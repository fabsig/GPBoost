# coding: utf-8
"""Exercise the Linux CPU quota code of GPBoost with injected file contents.

'CpuQuotaLimit()' in 'src/GPBoost/cpu_topology.cpp' limits the default number of threads by the CPU
bandwidth quota of the control group of the process. It reads '/proc/self/cgroup',
'/proc/self/mountinfo' and the quota files of the control group hierarchy, which cannot be arranged
freely on a real machine: a v1 hierarchy, a mount of a subtree, a covered mount or an escaped path
would each need a differently configured host. This script therefore compiles the actual
implementation against an in-memory filesystem and checks its decisions for such layouts.

Run it in Linux (a container is enough, the layouts are injected), with Python 3 and g++:
    python3 helpers/check_cpu_quota_fixtures.py .

The checkout is only read. A temporary copy of 'cpu_topology.cpp' redirects its 'ifstream' reads to
the injected files and 'sched_getaffinity()' to 16 synthetic CPUs, and OpenMP's initial maximum is
stubbed to 16. The quota parser, the mount resolver, the walk over the ancestors and the clamping of
the default number of threads are the ones of the checkout. This tests the quota decisions, not a
real kernel mount or a real OpenMP runtime.

A nonzero exit status means that an expectation failed or that the script could not run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile


def mount(root="/", point="/sys/fs/cgroup", version=2, controllers="cpu",
          mount_id=30, parent_id=20):
    """One line of '/proc/self/mountinfo'."""
    fs = "cgroup2" if version == 2 else "cgroup"
    options = "rw" if version == 2 else "rw," + controllers
    return f"{mount_id} {parent_id} 0:25 {root} {point} rw - {fs} cgroup {options}\n"


def v1_quota(files, directory, quota):
    """The two files in which a control group hierarchy v1 keeps a quota."""
    files[directory + "/cpu.cfs_quota_us"] = str(quota)
    files[directory + "/cpu.cfs_period_us"] = "100000"


def decode_mount_path(path):
    """Undoes the octal escapes that '/proc/self/mountinfo' writes for spaces and similar characters."""
    out, index = "", 0
    while index < len(path):
        if path[index] == chr(92) and path[index + 1:index + 4].isdigit():
            out += chr(int(path[index + 1:index + 4], 8))
            index += 4
        else:
            out += path[index]
            index += 1
    return out


def path_is_below(path, directory):
    """True if 'path' is 'directory' or lies below it."""
    if directory == "/":
        return True
    return path == directory or path.startswith(directory + "/")


def parse_mounts(mountinfo):
    """The entries of a '/proc/self/mountinfo' as dictionaries, in the order in which they are listed."""
    mounts = []
    for line in mountinfo.splitlines():
        if " - " not in line:
            continue
        fields = line.split(" - ")[0].split(" ")
        if len(fields) < 5:
            continue
        mounts.append({"id": int(fields[0]), "parent": int(fields[1]),
                       "root": decode_mount_path(fields[3]), "point": decode_mount_path(fields[4])})
    return mounts


def reachable_mounts(mounts):
    """
    The mounts that a path lookup can reach. A mount is covered by a later one at its own directory or
    at an ancestor of it, and a mount below a covered one cannot be reached either. A mount that covers
    its own parent stays reachable, since it is the one on top. This is the behaviour that the kernel
    reports through 'mnt_id', and it is modelled here so that the fixtures know the answer that the
    kernel would give.
    """
    identifiers = {mount["id"] for mount in mounts}
    reachable = {mount["id"]: True for mount in mounts}
    changed = True
    while changed:
        changed = False
        for index, mount in enumerate(mounts):
            if not reachable[mount["id"]]:
                continue
            covered = any(reachable[later["id"]] and later["id"] != mount["id"]
                          and path_is_below(mount["point"], later["point"])
                          for later in mounts[index + 1:])
            parent_ok = (mount["parent"] not in identifiers
                         or reachable[mount["parent"]]
                         or any(other["id"] == mount["parent"] and other["point"] == mount["point"]
                                for other in mounts))
            if covered or not parent_ok:
                reachable[mount["id"]] = False
                changed = True
    return reachable


def served_by_of(files):
    """For every injected file the identifier of the mount that the kernel would report for it."""
    mounts = parse_mounts(files.get("/proc/self/mountinfo", ""))
    reachable = reachable_mounts(mounts)
    served = {}
    for path in files:
        if path.startswith("/proc/"):
            continue
        serving = None
        for mount in mounts:
            if reachable[mount["id"]] and path_is_below(path, mount["point"]):
                if serving is None or len(mount["point"]) >= len(serving["point"]):
                    serving = mount
        if serving is not None:
            served[path] = serving["id"]
    return served


def fixture_cases():
    """The layouts to check, as (name, injected files, expected quota, expected default)."""
    cases = []

    def add(name, files, expected, default=None):
        cases.append((name, files.copy(), expected,
                      (min(16, expected) if expected else 16) if default is None else default,
                      served_by_of(files)))

    # The hierarchies are independent, so the controller has to be matched exactly: 'cpuacct' is not
    # the controller 'cpu' and its path must not be used for the quota
    files = {
        "/proc/self/cgroup": "8:cpu:/jobs/fit\n7:cpuacct:/jobs/accounting\n",
        "/proc/self/mountinfo": mount(point="/sys/fs/cgroup/cpu", version=1),
    }
    v1_quota(files, "/sys/fs/cgroup/cpu/jobs/fit", 400000)
    v1_quota(files, "/sys/fs/cgroup/cpu/jobs/accounting", 100000)
    add("v1 separate cpu and cpuacct", files, 4)
    files["/proc/self/cgroup"] = "7:cpuacct:/jobs/accounting\n8:cpu:/jobs/fit\n"
    add("v1 reversed membership rows", files, 4)
    for controllers in ("cpu,cpuset", "cpu,cpuacct"):
        files["/proc/self/cgroup"] = f"8:{controllers}:/jobs/fit\n"
        files["/proc/self/mountinfo"] = mount(
            point="/sys/fs/cgroup/cpu", version=1, controllers=controllers)
        add("v1 combined " + controllers, files, 4)

    # A mount can show a subtree of the hierarchy, so the path of the control group is relative to
    # the root of the mount and not to its mount point
    for version in (1, 2):
        files = {
            "/proc/self/cgroup": "0::/tenant/job\n" if version == 2
                                 else "8:cpu:/tenant/job\n",
            "/proc/self/mountinfo": mount(root="/tenant", point="/view/cgroup", version=version),
        }
        if version == 2:
            files["/view/cgroup/job/cpu.max"] = "400000 100000"
            files["/view/cgroup/tenant/job/cpu.max"] = "100000 100000"
        else:
            v1_quota(files, "/view/cgroup/job", 400000)
            v1_quota(files, "/view/cgroup/tenant/job", 100000)
        add(f"v{version} subtree with colliding path", files, 4)
        files["/proc/self/cgroup"] = "0::/tenant2/job\n" if version == 2 \
            else "8:cpu:/tenant2/job\n"
        add(f"v{version} group outside visible subtree", files, 0)

    # A quota of an ancestor applies as well, and the smallest one wins
    files = {"/proc/self/cgroup": "0::/jobs/fit\n",
             "/proc/self/mountinfo": mount()}
    files["/sys/fs/cgroup/jobs/fit/cpu.max"] = "400000 100000"
    files["/sys/fs/cgroup/jobs/cpu.max"] = "200000 100000"
    add("v2 tighter ancestor", files, 2)
    del files["/sys/fs/cgroup/jobs/cpu.max"]
    files["/sys/fs/cgroup/jobs/fit/cpu.max"] = "250000 100000"
    add("v2 fractional quota", files, 3)
    files["/sys/fs/cgroup/jobs/fit/cpu.max"] = "100000 100000"
    add("genuine one-CPU quota remains one", files, 1)
    files["/sys/fs/cgroup/jobs/fit/cpu.max"] = "max 100000"
    add("v2 unlimited", files, 0)

    files = {"/proc/self/cgroup": "8:cpu:/jobs/fit\n",
             "/proc/self/mountinfo": mount(point="/quota", version=1)}
    v1_quota(files, "/quota/jobs/fit", 400000)
    v1_quota(files, "/quota/jobs", 200000)
    add("v1 tighter ancestor at unusual mount", files, 2)

    # Above the mount point nothing is visible, so the quota there does not apply
    files = {"/proc/self/cgroup": "0::/tenant/job\n",
             "/proc/self/mountinfo": mount(root="/tenant", point="/view/cgroup"),
             "/view/cgroup/job/cpu.max": "400000 100000",
             "/view/cpu.max": "100000 100000"}
    add("ancestor walk stops at mount point", files, 4)

    files = {"/proc/self/cgroup": "7:cpuacct:/jobs/fit\n",
             "/proc/self/mountinfo": mount(point="/quota", version=1, controllers="cpuacct")}
    v1_quota(files, "/quota/jobs/fit", 100000)
    add("accounting controller alone is ignored", files, 0)

    # The first cgroup mount is covered by the second one at the same path. '/proc/self/mountinfo'
    # keeps both entries, but only the covering one can be read
    files = {"/proc/self/cgroup": "0::/tenant/job\n",
             "/proc/self/mountinfo": (
                 mount(mount_id=30, parent_id=20)
                 + mount(root="/tenant", mount_id=31, parent_id=30)),
             "/sys/fs/cgroup/job/cpu.max": "400000 100000",
             "/sys/fs/cgroup/tenant/job/cpu.max": "100000 100000"}
    add("stacked mounts must use visible root", files, 4)

    # A control group namespace reports '/../other' when the process has been moved outside the root
    # of its namespace. That group is not visible here, and removing the '..' would name another one
    files = {"/proc/self/cgroup": "0::/../other\n",
             "/proc/self/mountinfo": mount(),
             "/sys/fs/cgroup/cpu.max": "100000 100000"}
    add("outside namespace root must remain unknown", files, 0)

    # '/proc/self/mountinfo' escapes spaces, tabs, newlines and backslashes in its paths
    files = {"/proc/self/cgroup": "0::/job\n",
             "/proc/self/mountinfo": mount(point=r"/view/cgroup\040pool"),
             "/view/cgroup pool/job/cpu.max": "400000 100000"}
    add("escaped mount point", files, 4)
    files = {"/proc/self/cgroup": "0::/tenant pool/job\n",
             "/proc/self/mountinfo": mount(root=r"/tenant\040pool", point="/quota"),
             "/quota/job/cpu.max": "400000 100000"}
    add("escaped mount root", files, 4)

    # Both mounts are visible, and only the broader one shows the ancestor with the smaller quota
    files = {"/proc/self/cgroup": "0::/parent/tenant/job\n",
             "/proc/self/mountinfo": (
                 mount(root="/parent/tenant", point="/narrow")
                 + mount(point="/broad", mount_id=31)),
             "/narrow/job/cpu.max": "400000 100000",
             "/broad/parent/tenant/job/cpu.max": "400000 100000",
             "/broad/parent/cpu.max": "200000 100000"}
    add("broader visible mount exposes ancestor", files, 2)

    # A quota is only read where the mount that shows the control group really serves the directory.
    # Another mount can cover the hierarchy, redirect the directory of the control group, or redirect
    # one of its ancestors, and the control group found there is then a different one
    base_mount = "10 1 8:1 / / rw - ext4 /dev/root rw\n"
    for version in (1, 2):
        membership = "0::/tenant/job\n" if version == 2 else "8:cpu:/tenant/job\n"

        def set_quota(files, directory, value, version=version):
            if version == 2:
                files[directory + "/cpu.max"] = f"{value} 100000"
            else:
                v1_quota(files, directory, value)

        # A mount at '/view' covers the mount at '/view/cgroup' below it
        files = {"/proc/self/cgroup": membership,
                 "/proc/self/mountinfo": (
                     base_mount
                     + "20 10 0:26 / /view rw - tmpfs tmpfs rw\n"
                     + mount(point="/view/cgroup", version=version, mount_id=30, parent_id=20)
                     + mount(root="/tenant", point="/view", version=version, mount_id=31, parent_id=20))}
        set_quota(files, "/view/job", 400000)
        # Through the covering mount this would be the control group '/tenant/cgroup/tenant/job'
        set_quota(files, "/view/cgroup/tenant/job", 100000)
        add(f"v{version} mount hidden by an overmounted parent", files, 4)

        # The mount at '/cg' is visible, but another control group is mounted over the directory that
        # the control group of the process would have there. Only the view at '/clean' can be used
        files = {"/proc/self/cgroup": membership,
                 "/proc/self/mountinfo": (
                     base_mount
                     + mount(point="/cg", version=version, mount_id=30, parent_id=10)
                     + mount(root="/unrelated", point="/cg/tenant/job", version=version,
                             mount_id=31, parent_id=30)
                     + mount(point="/clean", version=version, mount_id=32, parent_id=10))}
        set_quota(files, "/cg/tenant/job", 100000)
        set_quota(files, "/clean/tenant/job", 400000)
        add(f"v{version} quota path redirected by a submount", files, 4)

    # The directory of the control group is correct, but an ancestor directory is served by a mount of
    # an unrelated control group, whose quota must not be taken for the one of an ancestor
    files = {"/proc/self/cgroup": "0::/tenant/job\n",
             "/proc/self/mountinfo": (
                 base_mount
                 + mount(point="/cg", mount_id=30, parent_id=10)
                 + mount(root="/unrelated", point="/cg/tenant", mount_id=31, parent_id=30)
                 + mount(root="/tenant/job", point="/cg/tenant/job", mount_id=32, parent_id=31)),
             "/cg/tenant/job/cpu.max": "400000 100000",
             "/cg/tenant/cpu.max": "100000 100000"}
    add("ancestor walk crosses into an unrelated mount", files, 4)

    # Mounts of other filesystems have to be taken into account as well: the file below the temporary
    # filesystem is ordinary data and not the quota of a control group
    files = {"/proc/self/cgroup": "0::/tenant/job\n",
             "/proc/self/mountinfo": (
                 base_mount
                 + mount(point="/cg", mount_id=30, parent_id=10)
                 + "31 30 0:26 / /cg rw - tmpfs tmpfs rw\n"
                 + mount(point="/clean", mount_id=32, parent_id=10)),
             "/cg/tenant/job/cpu.max": "100000 100000",
             "/clean/tenant/job/cpu.max": "400000 100000"}
    add("non-cgroup overmount hides a cgroup mount", files, 4)

    # A mount below a covered one cannot be reached either, although it is listed later and its
    # directory is the longest one that matches: its parent is the mount that has been covered
    files = {"/proc/self/cgroup": "0::/tenant/job\n",
             "/proc/self/mountinfo": (
                 base_mount
                 + "20 10 0:26 / /view rw - tmpfs tmpfs rw\n"
                 + mount(root="/tenant", point="/view", mount_id=30, parent_id=20)
                 + mount(root="/", point="/view/cg", mount_id=40, parent_id=20)),
             "/view/job/cpu.max": "400000 100000",
             "/view/cg/tenant/job/cpu.max": "100000 100000"}
    add("later child of a hidden parent is still unreachable", files, 4)
    return cases


IO_HEADER = r"""
#pragma once
#include <map>
#include <cstdlib>
#include <sched.h>
#include <sstream>
#include <string>
#include <vector>
extern std::map<std::string, std::string> review_files;
inline std::string ReviewNormalizePath(const std::string& path) {
    std::vector<std::string> parts;
    std::istringstream input(path);
    std::string part;
    while (std::getline(input, part, '/')) {
        if (part.empty() || part == ".") continue;
        if (part == "..") { if (!parts.empty()) parts.pop_back(); }
        else parts.push_back(part);
    }
    std::string result;
    for (const auto& item : parts) result += "/" + item;
    return result.empty() ? "/" : result;
}
extern std::map<std::string, int> review_served_by;
inline std::vector<std::string>& ReviewOpenPaths() {
    static std::vector<std::string> paths;
    return paths;
}
// The descriptors start at 3, as the first free one of a process
inline int ReviewOpen(const std::string& path) {
    const std::string normalized = ReviewNormalizePath(path);
    if (review_files.find(normalized) == review_files.end()) return -1;
    ReviewOpenPaths().push_back(normalized);
    return static_cast<int>(ReviewOpenPaths().size()) + 2;
}
inline long ReviewRead(int descriptor, char* buffer, size_t size) {
    const size_t index = static_cast<size_t>(descriptor) - 3;
    if (descriptor < 3 || index >= ReviewOpenPaths().size()) return -1;
    const std::string& content = review_files[ReviewOpenPaths()[index]];
    const size_t count = content.size() < size ? content.size() : size;
    content.copy(buffer, count);
    return static_cast<long>(count);
}
inline void ReviewClose(int) {}
// '/proc/self/fdinfo/<descriptor>' reports the mount that the kernel used for an open file
inline bool ReviewFdInfo(const std::string& path, std::string* content) {
    const std::string prefix = "/proc/self/fdinfo/";
    if (path.compare(0, prefix.size(), prefix) != 0) return false;
    const int descriptor = std::atoi(path.c_str() + prefix.size());
    const size_t index = static_cast<size_t>(descriptor) - 3;
    *content = "pos:\t0\nflags:\t0100000\n";
    if (descriptor >= 3 && index < ReviewOpenPaths().size()) {
        const auto found = review_served_by.find(ReviewOpenPaths()[index]);
        if (found != review_served_by.end()) {
            *content += "mnt_id:\t" + std::to_string(found->second) + "\n";
        }
    }
    return true;
}
class ReviewInputFile : public std::istringstream {
public:
    explicit ReviewInputFile(const std::string& path) {
        std::string content;
        if (ReviewFdInfo(path, &content)) {
            open_ = true;
            str(content);
            return;
        }
        const auto found = review_files.find(ReviewNormalizePath(path));
        open_ = found != review_files.end();
        if (open_) str(found->second);
        else setstate(std::ios::failbit);
    }
    bool is_open() const { return open_; }
private:
    bool open_;
};
inline int ReviewGetAffinity(pid_t, size_t size, cpu_set_t* mask) {
    CPU_ZERO_S(size, mask);
    for (int cpu = 0; cpu < 16; ++cpu) CPU_SET_S(cpu, size, mask);
    return 0;
}
"""

UTILS_HEADER = r"""
#pragma once
inline int omp_get_max_threads() { return 16; }
inline int omp_get_thread_limit() { return 16; }
namespace GPBoost {
    int CpuQuotaLimit();
    int ComputeDefaultNumParallelThreads();
}
"""


def driver_source(cases):
    """Source of a program that runs every case against the compiled implementation."""
    output = [r"""
#include "fixture_io.h"
#include <GPBoost/utils.h>
#include <iostream>
std::map<std::string, std::string> review_files;
std::map<std::string, int> review_served_by;
int main() {
    int failures = 0;
"""]
    for name, files, quota, default, served_by in cases:
        entries = ",\n".join(
            "{" + json.dumps(key) + ", " + json.dumps(value) + "}"
            for key, value in files.items())
        served = ",\n".join(
            "{" + json.dumps(key) + ", " + str(value) + "}"
            for key, value in served_by.items())
        output.append(f"""
    review_files = {{{entries}}};
    review_served_by = {{{served}}};
    {{
        const int quota = GPBoost::CpuQuotaLimit();
        const int threads = GPBoost::ComputeDefaultNumParallelThreads();
        const bool ok = quota == {quota} && threads == {default};
        std::cout << (ok ? "PASS " : "FAIL ") << {json.dumps(name)}
                  << " | quota=" << quota << " expected={quota}"
                  << " | default=" << threads << " expected={default}\\n";
        failures += !ok;
    }}
""")
    output.append(f"""
    std::cout << "{len(cases)} cases; " << failures << " failures\\n";
    return failures == 0 ? 0 : 1;
}}
""")
    return "".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, nargs="?", default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run this script for the Linux sources in Linux, in a container or in WSL.")
    if shutil.which(args.compiler) is None:
        parser.error(f"Compiler not found: {args.compiler}")
    source_path = args.checkout.resolve() / "src/GPBoost/cpu_topology.cpp"
    source_bytes = source_path.read_bytes()
    source = source_bytes.decode("utf-8-sig")
    if source.count("std::ifstream") != 4:
        parser.error("Source I/O changed: inspect this script before adapting its redirects.")
    affinity_call = "sched_getaffinity(0, mask_size, mask)"
    if source.count(affinity_call) != 1:
        parser.error("Source affinity call changed: inspect this script before adapting it.")
    print("Source:", source_path, flush=True)
    print("SHA256:", hashlib.sha256(source_bytes).hexdigest(), flush=True)
    source = '#include "fixture_io.h"\n' + source.replace("std::ifstream", "ReviewInputFile")
    source = source.replace(affinity_call, "ReviewGetAffinity(0, mask_size, mask)")
    for original, replacement in (("open(path.c_str(), O_RDONLY | O_CLOEXEC)", "ReviewOpen(path)"),
                                  ("read(descriptor, buffer, size)", "ReviewRead(descriptor, buffer, size)"),
                                  ("close(descriptor)", "ReviewClose(descriptor)")):
        if source.count(original) != 1:
            parser.error(f"Source file access changed: {original} is not used exactly once.")
        source = source.replace(original, replacement)
    with tempfile.TemporaryDirectory(prefix="gpboost-quota-fixtures-") as temporary:
        work = Path(temporary)
        (work / "GPBoost").mkdir()
        (work / "GPBoost/utils.h").write_text(UTILS_HEADER)
        (work / "fixture_io.h").write_text(IO_HEADER)
        (work / "cpu_topology.cpp").write_text(source)
        (work / "driver.cpp").write_text(driver_source(fixture_cases()))
        executable = work / "quota_fixtures"
        subprocess.run([args.compiler, "-std=c++11", "-Wall", "-Wextra", "-Werror",
                        "-I", str(work), str(work / "cpu_topology.cpp"),
                        str(work / "driver.cpp"), "-o", str(executable)], check=True)
        environment = os.environ.copy()
        # The fixture determines the default, an inherited explicit value would bypass it
        environment.pop("OMP_NUM_THREADS", None)
        return subprocess.run([str(executable)], env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
