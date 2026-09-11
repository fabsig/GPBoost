/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2025 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/utils.h>

#include <cstdlib>    // std::getenv

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

#include <sched.h>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
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
			for (int level = 0; level < num_performance_levels; ++level) {
				const std::string name = "hw.perflevel" + std::to_string(level) + ".physicalcpu";
				const int num_cores_of_level = SysctlInt(name.c_str());
				if (num_cores_of_level <= 0) {
					break;
				}
				num_cores += num_cores_of_level;
				if (num_cores >= MIN_NUM_CORES_OF_FASTEST_CLASSES) {
					break;
				}
			}
			if (num_cores > 0) {
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
	int CpuQuotaLimit() {
		// The path of the control group of this process relative to the mount point of the hierarchy. Every
		// line of '/proc/self/cgroup' is "hierarchy:controllers:path", where v2 uses the hierarchy 0 with an
		// empty list of controllers
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
			else if (controllers.find("cpu") != std::string::npos &&
				controllers.find("cpuset") == std::string::npos) {
				cgroup_path_v1 = path;
			}
		}
		// A quota of an ancestor also applies, so the smallest one of the control group and of all of its
		// ancestors is used. Only the usual mount points are looked at, '/proc/self/mountinfo' is not parsed
		double smallest_num_cpus = -1.;
		const char* v1_mount_points[] = { "/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct" };
		for (int hierarchy = 0; hierarchy < 3; ++hierarchy) {
			const bool is_v2 = hierarchy == 0;
			const std::string mount_point = is_v2 ? "/sys/fs/cgroup" : v1_mount_points[hierarchy - 1];
			std::string path = is_v2 ? cgroup_path_v2 : cgroup_path_v1;
			// Walk from the control group up to the root of the hierarchy
			while (true) {
				double quota = -1., period = -1.;
				const std::string directory = mount_point + path;
				std::string quota_string, period_string;
				if (is_v2) {
					// 'cpu.max' contains the quota and the period, and "max" means that there is no limit
					if (ReadSysFileLine(directory + "/cpu.max", &quota_string) &&
						quota_string.compare(0, 3, "max") != 0) {
						char* next = nullptr;
						quota = std::strtod(quota_string.c_str(), &next);
						if (next != quota_string.c_str()) {
							period = std::strtod(next, nullptr);
						}
					}
				}
				else if (ReadSysFileLine(directory + "/cpu.cfs_quota_us", &quota_string) &&
					ReadSysFileLine(directory + "/cpu.cfs_period_us", &period_string)) {
					// A negative quota means that there is no limit
					quota = std::strtod(quota_string.c_str(), nullptr);
					period = std::strtod(period_string.c_str(), nullptr);
				}
				if (quota > 0. && period > 0.) {
					const double num_cpus = std::ceil(quota / period);
					if (num_cpus >= 1. && (smallest_num_cpus < 0. || num_cpus < smallest_num_cpus)) {
						smallest_num_cpus = num_cpus;
					}
				}
				if (path.empty() || path == "/") {
					break;
				}
				const size_t last_slash = path.find_last_of('/');
				path = last_slash == std::string::npos ? std::string() : path.substr(0, last_slash);
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

	int ComputeDefaultNumParallelThreads() {
		int num_threads = omp_get_max_threads();
		// An explicitly requested number of threads is always used as is
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

}  // namespace GPBoost
