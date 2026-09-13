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


def fixture_cases():
    """The layouts to check, as (name, injected files, expected quota, expected default)."""
    cases = []

    def add(name, files, expected, default=None):
        cases.append((name, files.copy(), expected,
                      (min(16, expected) if expected else 16) if default is None else default))

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
    return cases


IO_HEADER = r"""
#pragma once
#include <map>
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
class ReviewInputFile : public std::istringstream {
public:
    explicit ReviewInputFile(const std::string& path) {
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
int main() {
    int failures = 0;
"""]
    for name, files, quota, default in cases:
        entries = ",\n".join(
            "{" + json.dumps(key) + ", " + json.dumps(value) + "}"
            for key, value in files.items())
        output.append(f"""
    review_files = {{{entries}}};
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
    if source.count("std::ifstream") != 3:
        parser.error("Source I/O changed: inspect this script before adapting its redirects.")
    affinity_call = "sched_getaffinity(0, mask_size, mask)"
    if source.count(affinity_call) != 1:
        parser.error("Source affinity call changed: inspect this script before adapting it.")
    print("Source:", source_path, flush=True)
    print("SHA256:", hashlib.sha256(source_bytes).hexdigest(), flush=True)
    source = '#include "fixture_io.h"\n' + source.replace("std::ifstream", "ReviewInputFile")
    source = source.replace(affinity_call, "ReviewGetAffinity(0, mask_size, mask)")
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
