# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dev/Documents/Elliot/MySpikingNetworks

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dev/Documents/Elliot/MySpikingNetworks

# Include any dependencies generated for this target.
include CMakeFiles/BaseExperiment.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/BaseExperiment.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/BaseExperiment.dir/flags.make

CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.o: CMakeFiles/BaseExperiment.dir/flags.make
CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.o: BaseExperiment.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dev/Documents/Elliot/MySpikingNetworks/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.o -c /home/dev/Documents/Elliot/MySpikingNetworks/BaseExperiment.cpp

CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dev/Documents/Elliot/MySpikingNetworks/BaseExperiment.cpp > CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.i

CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dev/Documents/Elliot/MySpikingNetworks/BaseExperiment.cpp -o CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.s

CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.o.requires:

.PHONY : CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.o.requires

CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.o.provides: CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.o.requires
	$(MAKE) -f CMakeFiles/BaseExperiment.dir/build.make CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.o.provides.build
.PHONY : CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.o.provides

CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.o.provides.build: CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.o


# Object files for target BaseExperiment
BaseExperiment_OBJECTS = \
"CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.o"

# External object files for target BaseExperiment
BaseExperiment_EXTERNAL_OBJECTS =

BaseExperiment: CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.o
BaseExperiment: CMakeFiles/BaseExperiment.dir/build.make
BaseExperiment: Spike/Build/Spike/libSpike.so
BaseExperiment: /usr/lib/x86_64-linux-gnu/libcudart_static.a
BaseExperiment: /usr/lib/x86_64-linux-gnu/librt.so
BaseExperiment: CMakeFiles/BaseExperiment.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dev/Documents/Elliot/MySpikingNetworks/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable BaseExperiment"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/BaseExperiment.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/BaseExperiment.dir/build: BaseExperiment

.PHONY : CMakeFiles/BaseExperiment.dir/build

CMakeFiles/BaseExperiment.dir/requires: CMakeFiles/BaseExperiment.dir/BaseExperiment.cpp.o.requires

.PHONY : CMakeFiles/BaseExperiment.dir/requires

CMakeFiles/BaseExperiment.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/BaseExperiment.dir/cmake_clean.cmake
.PHONY : CMakeFiles/BaseExperiment.dir/clean

CMakeFiles/BaseExperiment.dir/depend:
	cd /home/dev/Documents/Elliot/MySpikingNetworks && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dev/Documents/Elliot/MySpikingNetworks /home/dev/Documents/Elliot/MySpikingNetworks /home/dev/Documents/Elliot/MySpikingNetworks /home/dev/Documents/Elliot/MySpikingNetworks /home/dev/Documents/Elliot/MySpikingNetworks/CMakeFiles/BaseExperiment.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/BaseExperiment.dir/depend
