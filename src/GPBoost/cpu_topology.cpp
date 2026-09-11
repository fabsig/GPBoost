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
#include <type_traits>
#include <utility>
#include <vector>

#elif defined(__APPLE__)

#include <sys/sysctl.h>
#include <sys/types.h>
#include <cstddef>
#include <cstdint>

#elif defined(__linux__)

#include <sched.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <set>
#include <string>

#endif

namespace GPBoost {

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
	int EfficiencyClassOfCore(const T& core) {
		if constexpr (HasEfficiencyClass<T>::value) {
			return static_cast<int>(core.EfficiencyClass);
		}
		else {
			return static_cast<int>(core.Reserved[0]);
		}
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
		// Every 'RelationProcessorCore' record describes one physical core, i.e., simultaneous multithreading
		// siblings are counted only once. The first pass determines the efficiency class of the performance
		// cores, the second one counts them
		int highest_efficiency_class = -1;
		int num_cores = 0;
		for (int pass = 0; pass < 2; ++pass) {
			DWORD offset = 0;
			while (offset < buffer_size) {
				auto core_info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
				if (core_info->Size == 0) {
					return 0;
				}
				if (core_info->Relationship == RelationProcessorCore) {
					int efficiency_class = EfficiencyClassOfCore(core_info->Processor);
					if (pass == 0) {
						if (efficiency_class > highest_efficiency_class) {
							highest_efficiency_class = efficiency_class;
						}
					}
					else if (efficiency_class == highest_efficiency_class) {
						num_cores += 1;
					}
				}
				offset += core_info->Size;
			}
		}
		return num_cores;
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
		// On Apple silicon, performance level 0 is the highest-performing one
		if (SysctlInt("hw.nperflevels") > 1) {
			int num_cores = SysctlInt("hw.perflevel0.physicalcpu");
			if (num_cores > 0) {
				return num_cores;
			}
		}
		// Homogeneous CPU: physical cores, i.e., without the hyperthreading siblings of Intel Macs
		int num_physical_cores = SysctlInt("hw.physicalcpu");
		return num_physical_cores > 0 ? num_physical_cores : 0;
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
			if (first < 0 || last < first) {
				return false;
			}
			for (long cpu = first; cpu <= last; ++cpu) {
				cpus->insert(static_cast<int>(cpu));
			}
			if (*position != ',') {
				break;
			}
			position += 1;
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

	int NumPerformanceCores() {
		// Only CPUs this process may run on are considered
		std::set<int> cpus;
		cpu_set_t affinity;
		CPU_ZERO(&affinity);
		if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
			for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
				if (CPU_ISSET(cpu, &affinity)) {
					cpus.insert(cpu);
				}
			}
		}
		if (cpus.empty()) {
			std::string online_cpu_list;
			if (!ReadSysFileLine("/sys/devices/system/cpu/online", &online_cpu_list) ||
				!ParseCpuList(online_cpu_list, &cpus)) {
				return 0;
			}
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
				return CountPhysicalCores(selected_cpus);
			}
		}
		// Other heterogeneous CPUs (e.g., Arm big.LITTLE): 'cpu_capacity' gives the relative capacities of the
		// CPUs. It is used only if it is available for all of them
		long highest_capacity = -1;
		std::set<int> highest_capacity_cpus;
		for (int cpu : cpus) {
			std::string capacity_string;
			if (!ReadSysFileLine("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpu_capacity",
				&capacity_string)) {
				highest_capacity_cpus.clear();
				break;
			}
			char* next = nullptr;
			long capacity = std::strtol(capacity_string.c_str(), &next, 10);
			if (next == capacity_string.c_str() || capacity <= 0) {
				highest_capacity_cpus.clear();
				break;
			}
			if (capacity > highest_capacity) {
				highest_capacity = capacity;
				highest_capacity_cpus.clear();
			}
			if (capacity == highest_capacity) {
				highest_capacity_cpus.insert(cpu);
			}
		}
		if (!highest_capacity_cpus.empty()) {
			return CountPhysicalCores(highest_capacity_cpus);
		}
		// Homogeneous CPU, or no information on the core types: physical cores
		return CountPhysicalCores(cpus);
	}

#else

	int NumPerformanceCores() {
		return 0;
	}

#endif

	int ComputeDefaultNumParallelThreads() {
		int num_threads_omp = omp_get_max_threads();
		// An explicitly requested number of threads is always used as is
		const char* omp_num_threads_env = std::getenv("OMP_NUM_THREADS");
		if (omp_num_threads_env != nullptr && omp_num_threads_env[0] != '\0') {
			return num_threads_omp;
		}
		// The number of performance cores can only lower the number of threads, so that a number of threads
		// that is restricted by, e.g., a processor affinity mask is never exceeded
		int num_performance_cores = NumPerformanceCores();
		if (num_performance_cores > 0 && num_performance_cores < num_threads_omp) {
			return num_performance_cores;
		}
		return num_threads_omp;
	}

}  // namespace GPBoost
