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
include CMakeFiles/SimpleExample.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/SimpleExample.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SimpleExample.dir/flags.make

CMakeFiles/SimpleExample.dir/SimpleExample.cpp.o: CMakeFiles/SimpleExample.dir/flags.make
CMakeFiles/SimpleExample.dir/SimpleExample.cpp.o: SimpleExample.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dev/Documents/Elliot/MySpikingNetworks/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/SimpleExample.dir/SimpleExample.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SimpleExample.dir/SimpleExample.cpp.o -c /home/dev/Documents/Elliot/MySpikingNetworks/SimpleExample.cpp

CMakeFiles/SimpleExample.dir/SimpleExample.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SimpleExample.dir/SimpleExample.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dev/Documents/Elliot/MySpikingNetworks/SimpleExample.cpp > CMakeFiles/SimpleExample.dir/SimpleExample.cpp.i

CMakeFiles/SimpleExample.dir/SimpleExample.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SimpleExample.dir/SimpleExample.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dev/Documents/Elliot/MySpikingNetworks/SimpleExample.cpp -o CMakeFiles/SimpleExample.dir/SimpleExample.cpp.s

CMakeFiles/SimpleExample.dir/SimpleExample.cpp.o.requires:

.PHONY : CMakeFiles/SimpleExample.dir/SimpleExample.cpp.o.requires

CMakeFiles/SimpleExample.dir/SimpleExample.cpp.o.provides: CMakeFiles/SimpleExample.dir/SimpleExample.cpp.o.requires
	$(MAKE) -f CMakeFiles/SimpleExample.dir/build.make CMakeFiles/SimpleExample.dir/SimpleExample.cpp.o.provides.build
.PHONY : CMakeFiles/SimpleExample.dir/SimpleExample.cpp.o.provides

CMakeFiles/SimpleExample.dir/SimpleExample.cpp.o.provides.build: CMakeFiles/SimpleExample.dir/SimpleExample.cpp.o


# Object files for target SimpleExample
SimpleExample_OBJECTS = \
"CMakeFiles/SimpleExample.dir/SimpleExample.cpp.o"

# External object files for target SimpleExample
SimpleExample_EXTERNAL_OBJECTS =

SimpleExample: CMakeFiles/SimpleExample.dir/SimpleExample.cpp.o
SimpleExample: CMakeFiles/SimpleExample.dir/build.make
SimpleExample: Spike/Build/Spike/libSpike.so
SimpleExample: /usr/lib/x86_64-linux-gnu/libcudart_static.a
SimpleExample: /usr/lib/x86_64-linux-gnu/librt.so
SimpleExample: CMakeFiles/SimpleExample.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dev/Documents/Elliot/MySpikingNetworks/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable SimpleExample"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SimpleExample.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SimpleExample.dir/build: SimpleExample

.PHONY : CMakeFiles/SimpleExample.dir/build

CMakeFiles/SimpleExample.dir/requires: CMakeFiles/SimpleExample.dir/SimpleExample.cpp.o.requires

.PHONY : CMakeFiles/SimpleExample.dir/requires

CMakeFiles/SimpleExample.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SimpleExample.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SimpleExample.dir/clean

CMakeFiles/SimpleExample.dir/depend:
	cd /home/dev/Documents/Elliot/MySpikingNetworks && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dev/Documents/Elliot/MySpikingNetworks /home/dev/Documents/Elliot/MySpikingNetworks /home/dev/Documents/Elliot/MySpikingNetworks /home/dev/Documents/Elliot/MySpikingNetworks /home/dev/Documents/Elliot/MySpikingNetworks/CMakeFiles/SimpleExample.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/SimpleExample.dir/depend
