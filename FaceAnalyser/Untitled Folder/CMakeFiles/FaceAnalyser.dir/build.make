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
CMAKE_SOURCE_DIR = /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/build

# Include any dependencies generated for this target.
include CMakeFiles/FaceAnalyser.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/FaceAnalyser.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FaceAnalyser.dir/flags.make

CMakeFiles/FaceAnalyser.dir/src/Face_utils.o: CMakeFiles/FaceAnalyser.dir/flags.make
CMakeFiles/FaceAnalyser.dir/src/Face_utils.o: ../src/Face_utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/FaceAnalyser.dir/src/Face_utils.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FaceAnalyser.dir/src/Face_utils.o -c /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/Face_utils.cpp

CMakeFiles/FaceAnalyser.dir/src/Face_utils.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceAnalyser.dir/src/Face_utils.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/Face_utils.cpp > CMakeFiles/FaceAnalyser.dir/src/Face_utils.i

CMakeFiles/FaceAnalyser.dir/src/Face_utils.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceAnalyser.dir/src/Face_utils.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/Face_utils.cpp -o CMakeFiles/FaceAnalyser.dir/src/Face_utils.s

CMakeFiles/FaceAnalyser.dir/src/Face_utils.o.requires:

.PHONY : CMakeFiles/FaceAnalyser.dir/src/Face_utils.o.requires

CMakeFiles/FaceAnalyser.dir/src/Face_utils.o.provides: CMakeFiles/FaceAnalyser.dir/src/Face_utils.o.requires
	$(MAKE) -f CMakeFiles/FaceAnalyser.dir/build.make CMakeFiles/FaceAnalyser.dir/src/Face_utils.o.provides.build
.PHONY : CMakeFiles/FaceAnalyser.dir/src/Face_utils.o.provides

CMakeFiles/FaceAnalyser.dir/src/Face_utils.o.provides.build: CMakeFiles/FaceAnalyser.dir/src/Face_utils.o


CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.o: CMakeFiles/FaceAnalyser.dir/flags.make
CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.o: ../src/FaceAnalyser.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.o -c /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/FaceAnalyser.cpp

CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/FaceAnalyser.cpp > CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.i

CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/FaceAnalyser.cpp -o CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.s

CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.o.requires:

.PHONY : CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.o.requires

CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.o.provides: CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.o.requires
	$(MAKE) -f CMakeFiles/FaceAnalyser.dir/build.make CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.o.provides.build
.PHONY : CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.o.provides

CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.o.provides.build: CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.o


CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.o: CMakeFiles/FaceAnalyser.dir/flags.make
CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.o: ../src/SVM_dynamic_lin.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.o -c /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/SVM_dynamic_lin.cpp

CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/SVM_dynamic_lin.cpp > CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.i

CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/SVM_dynamic_lin.cpp -o CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.s

CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.o.requires:

.PHONY : CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.o.requires

CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.o.provides: CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.o.requires
	$(MAKE) -f CMakeFiles/FaceAnalyser.dir/build.make CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.o.provides.build
.PHONY : CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.o.provides

CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.o.provides.build: CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.o


CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.o: CMakeFiles/FaceAnalyser.dir/flags.make
CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.o: ../src/SVM_static_lin.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.o -c /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/SVM_static_lin.cpp

CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/SVM_static_lin.cpp > CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.i

CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/SVM_static_lin.cpp -o CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.s

CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.o.requires:

.PHONY : CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.o.requires

CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.o.provides: CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.o.requires
	$(MAKE) -f CMakeFiles/FaceAnalyser.dir/build.make CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.o.provides.build
.PHONY : CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.o.provides

CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.o.provides.build: CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.o


CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.o: CMakeFiles/FaceAnalyser.dir/flags.make
CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.o: ../src/SVR_dynamic_lin_regressors.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.o -c /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/SVR_dynamic_lin_regressors.cpp

CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/SVR_dynamic_lin_regressors.cpp > CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.i

CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/SVR_dynamic_lin_regressors.cpp -o CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.s

CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.o.requires:

.PHONY : CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.o.requires

CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.o.provides: CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.o.requires
	$(MAKE) -f CMakeFiles/FaceAnalyser.dir/build.make CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.o.provides.build
.PHONY : CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.o.provides

CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.o.provides.build: CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.o


CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.o: CMakeFiles/FaceAnalyser.dir/flags.make
CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.o: ../src/SVR_static_lin_regressors.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.o -c /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/SVR_static_lin_regressors.cpp

CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/SVR_static_lin_regressors.cpp > CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.i

CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/SVR_static_lin_regressors.cpp -o CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.s

CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.o.requires:

.PHONY : CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.o.requires

CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.o.provides: CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.o.requires
	$(MAKE) -f CMakeFiles/FaceAnalyser.dir/build.make CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.o.provides.build
.PHONY : CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.o.provides

CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.o.provides.build: CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.o


CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.o: CMakeFiles/FaceAnalyser.dir/flags.make
CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.o: ../src/GazeEstimation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.o -c /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/GazeEstimation.cpp

CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/GazeEstimation.cpp > CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.i

CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/src/GazeEstimation.cpp -o CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.s

CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.o.requires:

.PHONY : CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.o.requires

CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.o.provides: CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.o.requires
	$(MAKE) -f CMakeFiles/FaceAnalyser.dir/build.make CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.o.provides.build
.PHONY : CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.o.provides

CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.o.provides.build: CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.o


# Object files for target FaceAnalyser
FaceAnalyser_OBJECTS = \
"CMakeFiles/FaceAnalyser.dir/src/Face_utils.o" \
"CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.o" \
"CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.o" \
"CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.o" \
"CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.o" \
"CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.o" \
"CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.o"

# External object files for target FaceAnalyser
FaceAnalyser_EXTERNAL_OBJECTS =

libFaceAnalyser.a: CMakeFiles/FaceAnalyser.dir/src/Face_utils.o
libFaceAnalyser.a: CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.o
libFaceAnalyser.a: CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.o
libFaceAnalyser.a: CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.o
libFaceAnalyser.a: CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.o
libFaceAnalyser.a: CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.o
libFaceAnalyser.a: CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.o
libFaceAnalyser.a: CMakeFiles/FaceAnalyser.dir/build.make
libFaceAnalyser.a: CMakeFiles/FaceAnalyser.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX static library libFaceAnalyser.a"
	$(CMAKE_COMMAND) -P CMakeFiles/FaceAnalyser.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FaceAnalyser.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FaceAnalyser.dir/build: libFaceAnalyser.a

.PHONY : CMakeFiles/FaceAnalyser.dir/build

CMakeFiles/FaceAnalyser.dir/requires: CMakeFiles/FaceAnalyser.dir/src/Face_utils.o.requires
CMakeFiles/FaceAnalyser.dir/requires: CMakeFiles/FaceAnalyser.dir/src/FaceAnalyser.o.requires
CMakeFiles/FaceAnalyser.dir/requires: CMakeFiles/FaceAnalyser.dir/src/SVM_dynamic_lin.o.requires
CMakeFiles/FaceAnalyser.dir/requires: CMakeFiles/FaceAnalyser.dir/src/SVM_static_lin.o.requires
CMakeFiles/FaceAnalyser.dir/requires: CMakeFiles/FaceAnalyser.dir/src/SVR_dynamic_lin_regressors.o.requires
CMakeFiles/FaceAnalyser.dir/requires: CMakeFiles/FaceAnalyser.dir/src/SVR_static_lin_regressors.o.requires
CMakeFiles/FaceAnalyser.dir/requires: CMakeFiles/FaceAnalyser.dir/src/GazeEstimation.o.requires

.PHONY : CMakeFiles/FaceAnalyser.dir/requires

CMakeFiles/FaceAnalyser.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FaceAnalyser.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FaceAnalyser.dir/clean

CMakeFiles/FaceAnalyser.dir/depend:
	cd /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/build /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/build /home/olsen305/FACE/OpenFace-master/lib/local/FaceAnalyser/build/CMakeFiles/FaceAnalyser.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FaceAnalyser.dir/depend

