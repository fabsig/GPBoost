/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2025 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/utils.h>

#include <atomic>     // std::atomic
#include <cstdlib>    // std::getenv
#include <cstring>    // std::strcmp

#if defined(_MSC_VER)
#pragma warning( disable : 4996) // Suppress unnecessary warning ('getenv' is considered unsafe, but it is portable and the environment is only read here)
#endif

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

#elif defined(__APPLE__)

#include <sys/sysctl.h>
#include <sys/types.h>
#include <cstddef>
#include <cstdint>
#include <string>

#elif defined(__linux__)

#include <fcntl.h>
#include <sched.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

#endif

namespace GPBoost {

	/*!
	* \brief Smallest number of cores that the fastest classes of a heterogeneous CPU have to provide. CPUs with
	*		three or more classes exist whose fastest class contains a single core, e.g., Arm CPUs with one
	*		'prime' core, a few performance cores and several efficiency cores. Using that one core would make a
	*		multi-core machine single-threaded, so the next classes are added until this many cores are counted
	*/
	const int MIN_NUM_CORES_OF_FASTEST_CLASSES = 2;

#if defined(_WIN32)

	/*!
	* \brief True if 'PROCESSOR_RELATIONSHIP' has an 'EfficiencyClass' member. The headers of mingw-w64, which
	*		is used for the R package on Windows, declare the corresponding byte as 'Reserved[0]' instead
	*/
	template <typename T, typename = void>
	struct HasEfficiencyClass : std::false_type {};
	template <typename T>
	struct HasEfficiencyClass<T, decltype(void(std::declval<T>().EfficiencyClass))> : std::true_type {};

	/*!
	* \brief Efficiency class of a physical core. Larger values mean higher performance and lower efficiency,
	*		i.e., the performance cores of a hybrid CPU have the largest value. Windows always writes it to the
	*		byte that follows 'Flags', which is 'Reserved[0]' for headers that do not name it
	* \param core Description of a physical core as returned by 'GetLogicalProcessorInformationEx()'
	* \return Efficiency class of the core
	*/
	template <typename T>
	int EfficiencyClassOfCore(const T& core, std::true_type) {
		return static_cast<int>(core.EfficiencyClass);
	}
	template <typename T>
	int EfficiencyClassOfCore(const T& core, std::false_type) {
		return static_cast<int>(core.Reserved[0]);
	}
	// Note: this dispatches on the two overloads above instead of using 'if constexpr', since the R package
	// falls back to C++11 when the compiler does not support C++17, see 'R-package/configure.win'
	template <typename T>
	int EfficiencyClassOfCore(const T& core) {
		return EfficiencyClassOfCore(core, HasEfficiencyClass<T>());
	}

	/*!
	* \brief True if this process may run on at least one of the logical CPUs of a physical core
	* \param core Description of a physical core as returned by 'GetLogicalProcessorInformationEx()'
	* \param process_mask Affinity mask of this process, or 0 if it is not known
	* \return True if the core is available, and also if the affinity mask is not known
	*/
	template <typename T>
	bool CoreIsAvailable(const T& core, const DWORD_PTR process_mask) {
		if (process_mask == 0) {
			return true;
		}
		for (WORD group = 0; group < core.GroupCount; ++group) {
			if ((core.GroupMask[group].Mask & process_mask) != 0) {
				return true;
			}
		}
		return false;
	}

	int NumPerformanceCores() {
		DWORD buffer_size = 0;
		if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size) ||
			GetLastError() != ERROR_INSUFFICIENT_BUFFER || buffer_size == 0) {
			return 0;
		}
		std::vector<char> buffer(static_cast<size_t>(buffer_size));
		if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
			reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &buffer_size)) {
			return 0;
		}
		// The records describe the whole system, so the cores that this process may not run on have to be
		// excluded. The affinity mask of the process is used and not the one of the calling thread, which an
		// OpenMP runtime may have bound to a single core. 'GetProcessAffinityMask()' does not say which
		// processor group its mask belongs to, so it is used only if the system has a single group
		DWORD_PTR process_mask = 0, system_mask = 0;
		if (GetActiveProcessorGroupCount() != 1 ||
			!GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
			process_mask = 0;
		}
		// Every 'RelationProcessorCore' record describes one physical core, i.e., simultaneous multithreading
		// siblings are counted only once
		std::map<int, int> num_cores_per_efficiency_class;
		DWORD offset = 0;
		while (offset < buffer_size) {
			auto core_info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
			if (core_info->Size == 0) {
				return 0;
			}
			if (core_info->Relationship == RelationProcessorCore &&
				CoreIsAvailable(core_info->Processor, process_mask)) {
				num_cores_per_efficiency_class[EfficiencyClassOfCore(core_info->Processor)] += 1;
			}
			offset += core_info->Size;
		}
		// The cores of the highest efficiency class are used, and the next classes are added until enough cores
		// are counted, see 'MIN_NUM_CORES_OF_FASTEST_CLASSES'
		int num_cores = 0;
		for (auto efficiency_class = num_cores_per_efficiency_class.rbegin();
			efficiency_class != num_cores_per_efficiency_class.rend(); ++efficiency_class) {
			num_cores += efficiency_class->second;
			if (num_cores >= MIN_NUM_CORES_OF_FASTEST_CLASSES) {
				break;
			}
		}
		return num_cores;
	}

	/*! \brief Windows has no equivalent of the CPU bandwidth quota of the Linux control groups */
	int CpuQuotaLimit() {
		return 0;
	}

#elif defined(__APPLE__)

	/*!
	* \brief Reads an integer 'sysctl' value
	* \param name Name of the value
	* \return The value, or -1 if it is not available
	*/
	int SysctlInt(const char* name) {
		int64_t value = 0;
		size_t size = sizeof(value);
		if (sysctlbyname(name, &value, &size, nullptr, 0) != 0) {
			return -1;
		}
		return static_cast<int>(value);
	}

	int NumPerformanceCores() {
		// On Apple silicon, performance level 0 is the highest-performing one. The next levels are added
		// until enough cores are counted, see 'MIN_NUM_CORES_OF_FASTEST_CLASSES'
		const int num_performance_levels = SysctlInt("hw.nperflevels");
		if (num_performance_levels > 1) {
			int num_cores = 0;
			bool all_levels_read = true;
			for (int level = 0; level < num_performance_levels; ++level) {
				const std::string name = "hw.perflevel" + std::to_string(level) + ".physicalcpu";
				const int num_cores_of_level = SysctlInt(name.c_str());
				if (num_cores_of_level <= 0) {
					all_levels_read = false;
					break;
				}
				num_cores += num_cores_of_level;
				if (num_cores >= MIN_NUM_CORES_OF_FASTEST_CLASSES) {
					break;
				}
			}
			// A level that could not be read leaves too few cores, and the number of physical cores below is
			// then the better answer
			if (num_cores >= MIN_NUM_CORES_OF_FASTEST_CLASSES || (all_levels_read && num_cores > 0)) {
				return num_cores;
			}
		}
		// Homogeneous CPU: physical cores, i.e., without the hyperthreading siblings of Intel Macs
		int num_physical_cores = SysctlInt("hw.physicalcpu");
		return num_physical_cores > 0 ? num_physical_cores : 0;
	}

	/*! \brief macOS has no equivalent of the CPU bandwidth quota of the Linux control groups */
	int CpuQuotaLimit() {
		return 0;
	}

#elif defined(__linux__)

	/*!
	* \brief Reads the first line of a file in '/sys'
	* \param path Path of the file
	* \param[out] line First line of the file
	* \return True if a non-empty line has been read
	*/
	bool ReadSysFileLine(const std::string& path, std::string* line) {
		std::ifstream file(path);
		if (!file.is_open()) {
			return false;
		}
		if (!std::getline(file, *line)) {
			return false;
		}
		return !line->empty();
	}

	/*!
	* \brief Parses a CPU list such as "0-7,16,18-19" as written by the files in '/sys'
	* \param cpu_list The list
	* \param[out] cpus Numbers of the CPUs in the list
	* \return True if at least one CPU has been parsed and the list is well-formed
	*/
	bool ParseCpuList(const std::string& cpu_list, std::set<int>* cpus) {
		const char* position = cpu_list.c_str();
		while (*position != '\0') {
			char* next = nullptr;
			long first = std::strtol(position, &next, 10);
			if (next == position) {
				return false;
			}
			position = next;
			long last = first;
			if (*position == '-') {
				last = std::strtol(position + 1, &next, 10);
				if (next == position + 1) {
					return false;
				}
				position = next;
			}
			// The upper bound keeps a malformed or overflowing range from being expanded into a huge set
			if (first < 0 || last < first || last > (1 << 20)) {
				return false;
			}
			for (long cpu = first; cpu <= last; ++cpu) {
				cpus->insert(static_cast<int>(cpu));
			}
			if (*position == '\0') {
				break;
			}
			if (*position != ',') {
				return false;// trailing characters that are not a separator make the whole list invalid
			}
			position += 1;
			if (*position == '\0') {
				return false;// a separator has to be followed by another range
			}
		}
		return !cpus->empty();
	}

	/*!
	* \brief Counts the physical cores of a set of logical CPUs, i.e., counts simultaneous multithreading
	*		siblings only once
	* \param cpus Numbers of the logical CPUs
	* \return Number of physical cores
	*/
	int CountPhysicalCores(const std::set<int>& cpus) {
		std::set<int> cores;
		for (int cpu : cpus) {
			std::string topology_path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
			std::string sibling_list;
			std::set<int> siblings;
			// 'core_cpus_list' has been called 'thread_siblings_list' before Linux 5.3
			if ((ReadSysFileLine(topology_path + "core_cpus_list", &sibling_list) ||
				ReadSysFileLine(topology_path + "thread_siblings_list", &sibling_list)) &&
				ParseCpuList(sibling_list, &siblings)) {
				cores.insert(*siblings.begin());// the smallest sibling identifies the physical core
			}
			else {
				cores.insert(cpu);
			}
		}
		return static_cast<int>(cores.size());
	}

	/*!
	* \brief Numbers of the logical CPUs that the OpenMP threads of this process may run on
	* \param[out] cpus Numbers of the CPUs
	* \return True if the set of CPUs could be determined. If it could not, it must not be guessed: a set
	*		that is too large leads to more threads than there are cores, one that is too small to fewer
	*/
	bool AvailableCpus(std::set<int>* cpus) {
#if defined(_OPENMP) && _OPENMP >= 201307
		// If OpenMP binds its threads ('OMP_PROC_BIND'), the calling thread is already bound, possibly to a
		// single core, and its affinity mask is then not the set of CPUs that the whole team may use. The
		// OpenMP places describe that set and are used instead
		if (omp_get_proc_bind() != omp_proc_bind_false) {
#if _OPENMP >= 201511
			const int num_places = omp_get_num_places();
			for (int place = 0; place < num_places; ++place) {
				const int num_procs = omp_get_place_num_procs(place);
				if (num_procs <= 0) {
					continue;
				}
				std::vector<int> proc_ids(static_cast<size_t>(num_procs));
				omp_get_place_proc_ids(place, proc_ids.data());
				for (int proc_id : proc_ids) {
					if (proc_id >= 0) {
						cpus->insert(proc_id);
					}
				}
			}
#endif
			return !cpus->empty();
		}
#elif defined(_OPENMP)
		// OpenMP before 4.0 has neither 'omp_get_proc_bind()' nor places, but 'OMP_PROC_BIND' already binds
		// the threads since OpenMP 3.1. The affinity of the calling thread can then not be trusted, and
		// since there is no way to ask for the places, the available CPUs stay unknown
		const char* proc_bind_env = std::getenv("OMP_PROC_BIND");
		const char* places_env = std::getenv("OMP_PLACES");
		// 'GOMP_CPU_AFFINITY' of the GNU implementation binds the threads as well
		const char* gnu_affinity_env = std::getenv("GOMP_CPU_AFFINITY");
		const bool binding_requested =
			(proc_bind_env != nullptr && proc_bind_env[0] != '\0' &&
				std::strncmp(proc_bind_env, "false", 5) != 0 && std::strncmp(proc_bind_env, "FALSE", 5) != 0) ||
			(places_env != nullptr && places_env[0] != '\0') ||
			(gnu_affinity_env != nullptr && gnu_affinity_env[0] != '\0');
		if (binding_requested) {
			return false;
		}
#endif
		// 'cpu_set_t' has room for 'CPU_SETSIZE' (1024) CPUs only. A larger mask makes 'sched_getaffinity()'
		// fail with EINVAL and has to be allocated dynamically
		for (int num_cpus = CPU_SETSIZE; num_cpus <= (1 << 20); num_cpus *= 2) {
			cpu_set_t* mask = CPU_ALLOC(num_cpus);
			if (mask == nullptr) {
				return false;
			}
			const size_t mask_size = CPU_ALLOC_SIZE(num_cpus);
			CPU_ZERO_S(mask_size, mask);
			errno = 0;
			const int result = sched_getaffinity(0, mask_size, mask);
			const int error = errno;
			if (result == 0) {
				for (int cpu = 0; cpu < num_cpus; ++cpu) {
					if (CPU_ISSET_S(static_cast<size_t>(cpu), mask_size, mask)) {
						cpus->insert(cpu);
					}
				}
				CPU_FREE(mask);
				return !cpus->empty();
			}
			CPU_FREE(mask);
			if (error != EINVAL) {
				break;
			}
		}
		return false;
	}

	int NumPerformanceCores() {
		// Only CPUs this process may run on are considered
		std::set<int> cpus;
		if (!AvailableCpus(&cpus)) {
			return 0;
		}
		// On Intel hybrid CPUs, the 'cpu_core' performance monitoring unit lists the logical CPUs of the
		// performance cores (and 'cpu_atom' the ones of the efficiency cores)
		std::string performance_cpu_list;
		std::set<int> performance_cpus, selected_cpus;
		if (ReadSysFileLine("/sys/bus/event_source/devices/cpu_core/cpus", &performance_cpu_list) &&
			ParseCpuList(performance_cpu_list, &performance_cpus)) {
			std::set_intersection(cpus.begin(), cpus.end(), performance_cpus.begin(), performance_cpus.end(),
				std::inserter(selected_cpus, selected_cpus.begin()));
			if (!selected_cpus.empty()) {
				const int num_cores = CountPhysicalCores(selected_cpus);
				if (num_cores >= MIN_NUM_CORES_OF_FASTEST_CLASSES) {
					return num_cores;
				}
			}
		}
		// Other heterogeneous CPUs (e.g., Arm big.LITTLE): 'cpu_capacity' gives the relative capacities of the
		// CPUs. It is used only if it is available for all of them
		std::map<long, std::set<int>> cpus_per_capacity;
		for (int cpu : cpus) {
			std::string capacity_string;
			if (!ReadSysFileLine("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpu_capacity",
				&capacity_string)) {
				cpus_per_capacity.clear();
				break;
			}
			char* next = nullptr;
			long capacity = std::strtol(capacity_string.c_str(), &next, 10);
			if (next == capacity_string.c_str() || capacity <= 0) {
				cpus_per_capacity.clear();
				break;
			}
			cpus_per_capacity[capacity].insert(cpu);
		}
		if (!cpus_per_capacity.empty()) {
			// The fastest CPUs are used, but the next classes are added until enough cores are counted, see
			// 'MIN_NUM_CORES_OF_FASTEST_CLASSES'
			std::set<int> selected;
			int num_cores = 0;
			for (auto capacity = cpus_per_capacity.rbegin(); capacity != cpus_per_capacity.rend(); ++capacity) {
				selected.insert(capacity->second.begin(), capacity->second.end());
				num_cores = CountPhysicalCores(selected);
				if (num_cores >= MIN_NUM_CORES_OF_FASTEST_CLASSES) {
					break;
				}
			}
			return num_cores;
		}
		// Homogeneous CPU, or no information on the core types: physical cores
		return CountPhysicalCores(cpus);
	}

	/*!
	* \brief Number of CPUs that the bandwidth quota of the control group of this process corresponds to. A
	*		quota (e.g., 'docker run --cpus=8') limits the CPU time per period and, unlike a cpuset, does not
	*		show up in the processor affinity. More threads than this only make the group be throttled
	* \return Number of CPUs, or 0 if there is no quota or it cannot be determined
	*/
	/*!
	* \brief Splits a string at a delimiter
	* \param text The string
	* \param delimiter The delimiter
	* \return The parts between the delimiters
	*/
	std::vector<std::string> SplitString(const std::string& text, const char delimiter) {
		std::vector<std::string> parts;
		size_t start = 0;
		while (true) {
			const size_t position = text.find(delimiter, start);
			parts.push_back(text.substr(start, position == std::string::npos ? std::string::npos : position - start));
			if (position == std::string::npos) {
				return parts;
			}
			start = position + 1;
		}
	}

	/*!
	* \brief True if a comma-separated list contains a token. The token has to match completely: the list of
	*		control group controllers "cpuacct" does not contain the controller "cpu"
	* \param list Comma-separated list
	* \param token The token
	* \return True if the list contains the token
	*/
	bool ListContainsToken(const std::string& list, const std::string& token) {
		const std::vector<std::string> tokens = SplitString(list, ',');
		return std::find(tokens.begin(), tokens.end(), token) != tokens.end();
	}

	/*!
	* \brief Decodes the octal escapes that '/proc/self/mountinfo' writes for spaces, tabs, newlines and
	*		backslashes in paths
	* \param path Path as written in '/proc/self/mountinfo'
	* \return The path with the escapes replaced by the characters they stand for
	*/
	std::string DecodeMountPath(const std::string& path) {
		std::string decoded;
		for (size_t position = 0; position < path.size(); ++position) {
			const bool is_escape = path[position] == '\\' && position + 3 < path.size() &&
				path[position + 1] >= '0' && path[position + 1] <= '7' &&
				path[position + 2] >= '0' && path[position + 2] <= '7' &&
				path[position + 3] >= '0' && path[position + 3] <= '7';
			if (is_escape) {
				const int value = (path[position + 1] - '0') * 64 + (path[position + 2] - '0') * 8 +
					(path[position + 3] - '0');
				decoded.push_back(static_cast<char>(value));
				position += 3;
			}
			else {
				decoded.push_back(path[position]);
			}
		}
		return decoded;
	}

	/*!
	* \brief True if a path contains a '..' component. A control group namespace writes such a path when the
	*		process has been moved outside the root of its namespace. The group is then not visible, and the
	*		path must not be resolved: removing the component would name a different group
	* \param path The path
	* \return True if the path contains a '..' component
	*/
	bool PathHasParentComponent(const std::string& path) {
		const std::vector<std::string> components = SplitString(path, '/');
		return std::find(components.begin(), components.end(), std::string("..")) != components.end();
	}

	/*! \brief One entry of '/proc/self/mountinfo' */
	struct MountEntry {
		/*! \brief Identifier of the mount, which the kernel also reports for an open file of it */
		int id;
		/*! \brief Directory of the mounted filesystem that is shown, '/' for all of it */
		std::string root;
		/*! \brief Directory at which it is shown */
		std::string point;
		/*! \brief Type of the filesystem, e.g. "cgroup2" */
		std::string filesystem;
		/*! \brief Options of the filesystem, which list the controllers of a v1 control group hierarchy */
		std::string super_options;
	};

	/*!
	* \brief Opens a file for reading. Wrapped in a function of its own so that the tests can replace it
	* \param path Path of the file
	* \return Descriptor of the open file, or a negative number if it could not be opened
	*/
	int OpenFileForReading(const std::string& path) {
		return open(path.c_str(), O_RDONLY | O_CLOEXEC);
	}

	/*!
	* \brief Reads from an open file. Wrapped in a function of its own so that the tests can replace it
	* \param descriptor Descriptor of the open file
	* \param[out] buffer Buffer to read into
	* \param size Size of the buffer
	* \return Number of bytes read, or a negative number if the file could not be read
	*/
	long ReadFromFile(int descriptor, char* buffer, size_t size) {
		return static_cast<long>(read(descriptor, buffer, size));
	}

	/*! \brief Closes an open file. Wrapped in a function of its own so that the tests can replace it */
	void CloseFile(int descriptor) {
		close(descriptor);
	}

	/*!
	* \brief Reads the first line of a file, but only if the kernel confirms that the file is served by a
	*		given mount. A directory of a control group hierarchy can be covered by another mount, which makes
	*		it show a different control group, and which mount serves a path cannot be determined reliably from
	*		the paths in '/proc/self/mountinfo' alone. The kernel reports the mount of an open file as 'mnt_id'
	*		in '/proc/self/fdinfo', and only the file that this identifies is used
	* \param path Path of the file
	* \param mount_id Identifier of the mount that has to serve the file
	* \param[out] line First line of the file
	* \return True if the file is served by the mount and a non-empty line has been read
	*/
	bool ReadFileFromMount(const std::string& path,
		const int mount_id,
		std::string* line) {
		const int descriptor = OpenFileForReading(path);
		if (descriptor < 0) {
			return false;
		}
		bool serves = false;
		std::string fdinfo_line;
		std::ifstream fdinfo_file("/proc/self/fdinfo/" + std::to_string(descriptor));
		const std::string key = "mnt_id:";
		while (std::getline(fdinfo_file, fdinfo_line)) {
			if (fdinfo_line.compare(0, key.size(), key) == 0) {
				serves = std::atoi(fdinfo_line.c_str() + key.size()) == mount_id;
				break;
			}
		}
		bool read_line = false;
		if (serves) {
			// The quota files hold a few numbers only
			char buffer[256];
			const long num_bytes = ReadFromFile(descriptor, buffer, sizeof(buffer) - 1);
			if (num_bytes > 0) {
				buffer[num_bytes] = '\0';
				const std::string content(buffer);
				const size_t newline = content.find('\n');
				*line = newline == std::string::npos ? content : content.substr(0, newline);
				read_line = !line->empty();
			}
		}
		CloseFile(descriptor);
		return read_line;
	}

	/*!
	* \brief True if a path is a directory or lies below it
	* \param path The path
	* \param directory The directory
	* \return True if 'path' is 'directory' or below it
	*/
	bool PathIsBelow(const std::string& path, const std::string& directory) {
		if (directory == "/") {
			return true;
		}
		if (path.size() < directory.size() || path.compare(0, directory.size(), directory) != 0) {
			return false;
		}
		return path.size() == directory.size() || path[directory.size()] == '/';
	}

	/*! \brief All entries of '/proc/self/mountinfo', in the order in which they are listed */
	std::vector<MountEntry> ReadMountEntries() {
		std::vector<MountEntry> entries;
		std::ifstream mountinfo_file("/proc/self/mountinfo");
		std::string line;
		while (std::getline(mountinfo_file, line)) {
			// "id parent major:minor root mount_point options... - filesystem source super_options"
			const size_t separator = line.find(" - ");
			if (separator == std::string::npos) {
				continue;
			}
			const std::vector<std::string> fields = SplitString(line.substr(0, separator), ' ');
			const std::vector<std::string> filesystem_fields = SplitString(line.substr(separator + 3), ' ');
			if (fields.size() < 5 || filesystem_fields.size() < 3) {
				continue;
			}
			MountEntry entry;
			entry.id = std::atoi(fields[0].c_str());
			entry.root = DecodeMountPath(fields[3]);
			entry.point = DecodeMountPath(fields[4]);
			entry.filesystem = filesystem_fields[0];
			entry.super_options = filesystem_fields[2];
			entries.push_back(entry);
		}
		return entries;
	}

	/*! \brief A control group as it can be read through one mount of its hierarchy */
	struct CgroupMapping {
		/*! \brief Directory of the control group */
		std::string directory;
		/*! \brief Directory at which the hierarchy is mounted, i.e., the highest ancestor visible here */
		std::string mount_point;
		/*! \brief Identifier of the mount through which the control group is read */
		int mount_id;
	};

	/*!
	* \brief Directories in which the control group of this process can be read
	* \param cgroup_path Path of the control group relative to the root of its hierarchy, as written in
	*		'/proc/self/cgroup'
	* \param version_2 True for the control group hierarchy v2, false for the v1 hierarchy of the controller 'cpu'
	* \return One entry per mount through which the control group is visible, empty if it cannot be located.
	*		A mount can show a subtree of the hierarchy, so the path from '/proc/self/cgroup' must not simply
	*		be appended to the mount point: it is relative to the root of the hierarchy, while the mount shows
	*		the subtree below the root of the mount. Different mounts can also make different ancestors
	*		readable, so all of them are returned
	*/
	std::vector<CgroupMapping> ResolveCgroupDirectories(const std::vector<MountEntry>& entries,
		const std::string& cgroup_path,
		const bool version_2) {
		std::vector<CgroupMapping> mappings;
		if (cgroup_path.empty() || PathHasParentComponent(cgroup_path)) {
			return mappings;
		}
		for (size_t index = 0; index < entries.size(); ++index) {
			const MountEntry& entry = entries[index];
			if (version_2 ? entry.filesystem != "cgroup2"
				: (entry.filesystem != "cgroup" || !ListContainsToken(entry.super_options, "cpu"))) {
				continue;
			}
			// Only the part of the hierarchy below the root of the mount is visible
			std::string relative_path = cgroup_path;
			if (entry.root != "/") {
				if (!PathIsBelow(cgroup_path, entry.root)) {
					continue;
				}
				relative_path = cgroup_path.substr(entry.root.size());
			}
			CgroupMapping mapping;
			mapping.mount_point = entry.point;
			mapping.directory = entry.point + (relative_path == "/" ? std::string() : relative_path);
			mapping.mount_id = entry.id;
			// Whether this mount really serves the directory is confirmed by the kernel when the quota is
			// read, see 'ReadFileFromMount'
			mappings.push_back(mapping);
		}
		return mappings;
	}

	/*!
	* \brief Reads the CPU bandwidth quota of a control group and keeps it if it is the smallest one so far
	* \param directory Directory of the control group
	* \param version_2 True for the control group hierarchy v2
	* \param mount_id Identifier of the mount that has to serve the files
	* \param[out] smallest_num_cpus Smallest number of CPUs found so far, negative if there is none
	*/
	void UpdateSmallestQuota(const std::string& directory,
		const bool version_2,
		const int mount_id,
		double* smallest_num_cpus) {
		double quota = -1., period = -1.;
		std::string quota_string, period_string;
		if (version_2) {
			// 'cpu.max' contains the quota and the period, and "max" means that there is no limit
			if (ReadFileFromMount(directory + "/cpu.max", mount_id, &quota_string) &&
				quota_string.compare(0, 3, "max") != 0) {
				char* next = nullptr;
				quota = std::strtod(quota_string.c_str(), &next);
				if (next != quota_string.c_str()) {
					period = std::strtod(next, nullptr);
				}
			}
		}
		else if (ReadFileFromMount(directory + "/cpu.cfs_quota_us", mount_id, &quota_string) &&
			ReadFileFromMount(directory + "/cpu.cfs_period_us", mount_id, &period_string)) {
			// A negative quota means that there is no limit
			quota = std::strtod(quota_string.c_str(), nullptr);
			period = std::strtod(period_string.c_str(), nullptr);
		}
		if (quota > 0. && period > 0.) {
			const double num_cpus = std::ceil(quota / period);
			if (num_cpus >= 1. && (*smallest_num_cpus < 0. || num_cpus < *smallest_num_cpus)) {
				*smallest_num_cpus = num_cpus;
			}
		}
	}

	int CpuQuotaLimit() {
		// The path of the control group of this process relative to the root of its hierarchy. Every line of
		// '/proc/self/cgroup' is "hierarchy:controllers:path", where v2 uses the hierarchy 0 with an empty
		// list of controllers. The hierarchies are independent, so 'cpu' and, e.g., 'cpuacct' can have
		// different paths and the controller has to be matched exactly
		std::string cgroup_path_v1, cgroup_path_v2, line;
		std::ifstream cgroup_file("/proc/self/cgroup");
		while (std::getline(cgroup_file, line)) {
			const size_t first_colon = line.find(':');
			if (first_colon == std::string::npos) {
				continue;
			}
			const size_t second_colon = line.find(':', first_colon + 1);
			if (second_colon == std::string::npos) {
				continue;
			}
			const std::string controllers = line.substr(first_colon + 1, second_colon - first_colon - 1);
			const std::string path = line.substr(second_colon + 1);
			if (controllers.empty()) {
				cgroup_path_v2 = path;
			}
			else if (ListContainsToken(controllers, "cpu")) {
				cgroup_path_v1 = path;
			}
		}
		// A quota of an ancestor also applies, so the smallest one of the control group and of all of its
		// visible ancestors is used
		const std::vector<MountEntry> entries = ReadMountEntries();
		double smallest_num_cpus = -1.;
		for (int version = 2; version >= 1; --version) {
			const bool version_2 = version == 2;
			const std::vector<CgroupMapping> mappings = ResolveCgroupDirectories(
				entries, version_2 ? cgroup_path_v2 : cgroup_path_v1, version_2);
			for (size_t i = 0; i < mappings.size(); ++i) {
				const std::string& mount_point = mappings[i].mount_point;
				std::string directory = mappings[i].directory;
				while (true) {
					// A directory can be covered by another mount and then belong to a different control
					// group. Only a file that the kernel reports as served by this mount is used
					UpdateSmallestQuota(directory, version_2, mappings[i].mount_id, &smallest_num_cpus);
					if (directory.size() <= mount_point.size()) {
						break;
					}
					const size_t last_slash = directory.find_last_of('/');
					directory = (last_slash == std::string::npos || last_slash < mount_point.size())
						? mount_point : directory.substr(0, last_slash);
				}
			}
		}
		if (!(smallest_num_cpus >= 1.) || smallest_num_cpus > (double)(1 << 20)) {
			return 0;
		}
		return static_cast<int>(smallest_num_cpus);
	}

#else

	int NumPerformanceCores() {
		return 0;
	}

	int CpuQuotaLimit() {
		return 0;
	}

#endif

	namespace {

		/*!
		* \brief Largest number of threads that OMP can run. The number of threads of a parallel region is
		*		limited both by the number of threads that OMP would use ('omp_get_max_threads()', usually the
		*		number of logical processors or the value of 'OMP_NUM_THREADS') and by the limit of the
		*		contention group ('omp_get_thread_limit()', the value of 'OMP_THREAD_LIMIT'). The two are
		*		different internal control variables, so the first one alone can report more threads than OMP
		*		would ever create
		* \return Largest number of threads that OMP can run
		*/
		int OmpThreadCeiling() {
			int num_threads = omp_get_max_threads();
			const int num_threads_contention_group = omp_get_thread_limit();
			if (num_threads_contention_group > 0 && num_threads_contention_group < num_threads) {
				num_threads = num_threads_contention_group;
			}
			return num_threads;
		}

		/*! \brief Determines the automatically selected number of threads from an already read number of threads of OMP */
		int ComputeAutoNumParallelThreadsFrom(int num_threads) {
			// An explicitly requested number of threads is used as it is, i.e. neither the cores nor a
			// quota lower it. The limits of the OpenMP runtime have already been applied to it
			const char* omp_num_threads_env = std::getenv("OMP_NUM_THREADS");
			if (omp_num_threads_env != nullptr && omp_num_threads_env[0] != '\0') {
				return num_threads;
			}
			// The number of performance cores and the quota can only lower the number of threads, so that a
			// number of threads that is restricted by, e.g., a processor affinity mask is never exceeded
			const int num_performance_cores = NumPerformanceCores();
			if (num_performance_cores > 0 && num_performance_cores < num_threads) {
				num_threads = num_performance_cores;
			}
			const int num_cpus_quota = CpuQuotaLimit();
			if (num_cpus_quota > 0 && num_cpus_quota < num_threads) {
				num_threads = num_cpus_quota;
			}
			return num_threads;
		}

	}  // namespace

	int ComputeDefaultNumParallelThreads() {
		return ComputeAutoNumParallelThreadsFrom(OmpThreadCeiling());
	}

	namespace {

		/*! \brief True if the number of threads has been requested explicitly via the environment */
		bool OmpNumThreadsEnvIsSet() {
			const char* omp_num_threads_env = std::getenv("OMP_NUM_THREADS");
			return omp_num_threads_env != nullptr && omp_num_threads_env[0] != '\0';
		}

		/*! \brief True if the message about the automatically selected number of threads has been switched off */
		bool AutoNumParallelThreadsMessageIsDisabled() {
			const char* message_env = std::getenv("GPBOOST_THREAD_MESSAGE");
			if (message_env == nullptr || message_env[0] == '\0') {
				return false;
			}
			return std::strcmp(message_env, "0") == 0 || std::strcmp(message_env, "false") == 0 ||
				std::strcmp(message_env, "FALSE") == 0;
		}

		/*! \brief The numbers of threads that are determined once, from a single number of threads of OMP */
		struct NumParallelThreadsDefaults {
			/*! \brief Automatically selected number of threads */
			int automatic;
			/*! \brief Largest number of threads that is used on its own */
			int maximum;
			/*! \brief True if the number of threads has been requested explicitly via the environment */
			bool omp_num_threads_env_is_set;
		};

		/*!
		* \brief The numbers of threads that are derived from the machine. They are determined when this function is
		*		called for the first time, which must happen before the library has changed the number of threads
		*		of the process: the number of threads of OMP is the upper limit for both of them
		* \return The numbers of threads that are derived from the machine
		*/
		const NumParallelThreadsDefaults& NumParallelThreadsDefaultsOfMachine() {
			static const NumParallelThreadsDefaults defaults = []() {
				NumParallelThreadsDefaults values;
				const int num_threads_omp = OmpThreadCeiling();
				values.omp_num_threads_env_is_set = OmpNumThreadsEnvIsSet();
				values.automatic = ComputeAutoNumParallelThreadsFrom(num_threads_omp);
				// The largest number of threads is not limited by the number of performance cores: a
				// benchmark can find that the slower cores or the hyperthreads help. A CPU bandwidth limit
				// of a control group is a limit of the machine and not a property of the cores, so it
				// applies here as well, unless the number of threads has been requested explicitly
				values.maximum = num_threads_omp;
				if (!values.omp_num_threads_env_is_set) {
					const int num_cpus_quota = CpuQuotaLimit();
					if (num_cpus_quota > 0 && num_cpus_quota < values.maximum) {
						values.maximum = num_cpus_quota;
					}
				}
				if (values.maximum < values.automatic) {
					values.maximum = values.automatic;
				}
				return values;
			}();
			return defaults;
		}

		/*! \brief The number of threads that has been set for the session, or 0 if there is none */
		std::atomic<int>& TunedNumParallelThreadsStorage() {
			static std::atomic<int> num_threads_tuned(0);
			return num_threads_tuned;
		}

		/*! \brief True if the message about the automatically selected number of threads has already been written */
		std::atomic<bool>& AutoNumParallelThreadsMessageIsDone() {
			// A number of threads that has been requested explicitly via the environment is a decision of the user:
			//	the message is not written in that case, and thus also not when the message is switched off
			static std::atomic<bool> is_done(NumParallelThreadsDefaultsOfMachine().omp_num_threads_env_is_set ||
				AutoNumParallelThreadsMessageIsDisabled());
			return is_done;
		}

	}  // namespace

	int AutoNumParallelThreads() {
		return NumParallelThreadsDefaultsOfMachine().automatic;
	}

	int MaxNumParallelThreads() {
		return NumParallelThreadsDefaultsOfMachine().maximum;
	}

	int TunedNumParallelThreads() {
		return TunedNumParallelThreadsStorage().load(std::memory_order_relaxed);
	}

	void SetDefaultNumParallelThreads(int num_threads) {
		// The numbers of threads of the machine are determined before the default is changed, and also when the
		//	default is only reset here: they must not be derived from a number of threads that the library has set
		const NumParallelThreadsDefaults& defaults = NumParallelThreadsDefaultsOfMachine();
		int num_threads_used = num_threads;
		if (num_threads_used > defaults.maximum) {
			num_threads_used = defaults.maximum;
		}
		if (num_threads_used < 0) {
			num_threads_used = 0;
		}
		TunedNumParallelThreadsStorage().store(num_threads_used, std::memory_order_relaxed);
	}

	bool ClaimAutoNumParallelThreadsMessage() {
		std::atomic<bool>& is_done = AutoNumParallelThreadsMessageIsDone();
		if (is_done.load(std::memory_order_relaxed)) {
			return false;
		}
		return !is_done.exchange(true, std::memory_order_relaxed);
	}

	void SuppressAutoNumParallelThreadsMessage() {
		AutoNumParallelThreadsMessageIsDone().store(true, std::memory_order_relaxed);
	}

}  // namespace GPBoost
