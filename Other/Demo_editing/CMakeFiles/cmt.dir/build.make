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
CMAKE_SOURCE_DIR = /home/gerrywalsh/Videos/Demo_editing

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/gerrywalsh/Videos/Demo_editing

# Include any dependencies generated for this target.
include CMakeFiles/cmt.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cmt.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cmt.dir/flags.make

CMakeFiles/cmt.dir/common.cpp.o: CMakeFiles/cmt.dir/flags.make
CMakeFiles/cmt.dir/common.cpp.o: common.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gerrywalsh/Videos/Demo_editing/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cmt.dir/common.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cmt.dir/common.cpp.o -c /home/gerrywalsh/Videos/Demo_editing/common.cpp

CMakeFiles/cmt.dir/common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cmt.dir/common.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gerrywalsh/Videos/Demo_editing/common.cpp > CMakeFiles/cmt.dir/common.cpp.i

CMakeFiles/cmt.dir/common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cmt.dir/common.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gerrywalsh/Videos/Demo_editing/common.cpp -o CMakeFiles/cmt.dir/common.cpp.s

CMakeFiles/cmt.dir/common.cpp.o.requires:

.PHONY : CMakeFiles/cmt.dir/common.cpp.o.requires

CMakeFiles/cmt.dir/common.cpp.o.provides: CMakeFiles/cmt.dir/common.cpp.o.requires
	$(MAKE) -f CMakeFiles/cmt.dir/build.make CMakeFiles/cmt.dir/common.cpp.o.provides.build
.PHONY : CMakeFiles/cmt.dir/common.cpp.o.provides

CMakeFiles/cmt.dir/common.cpp.o.provides.build: CMakeFiles/cmt.dir/common.cpp.o


CMakeFiles/cmt.dir/gui.cpp.o: CMakeFiles/cmt.dir/flags.make
CMakeFiles/cmt.dir/gui.cpp.o: gui.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gerrywalsh/Videos/Demo_editing/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/cmt.dir/gui.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cmt.dir/gui.cpp.o -c /home/gerrywalsh/Videos/Demo_editing/gui.cpp

CMakeFiles/cmt.dir/gui.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cmt.dir/gui.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gerrywalsh/Videos/Demo_editing/gui.cpp > CMakeFiles/cmt.dir/gui.cpp.i

CMakeFiles/cmt.dir/gui.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cmt.dir/gui.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gerrywalsh/Videos/Demo_editing/gui.cpp -o CMakeFiles/cmt.dir/gui.cpp.s

CMakeFiles/cmt.dir/gui.cpp.o.requires:

.PHONY : CMakeFiles/cmt.dir/gui.cpp.o.requires

CMakeFiles/cmt.dir/gui.cpp.o.provides: CMakeFiles/cmt.dir/gui.cpp.o.requires
	$(MAKE) -f CMakeFiles/cmt.dir/build.make CMakeFiles/cmt.dir/gui.cpp.o.provides.build
.PHONY : CMakeFiles/cmt.dir/gui.cpp.o.provides

CMakeFiles/cmt.dir/gui.cpp.o.provides.build: CMakeFiles/cmt.dir/gui.cpp.o


CMakeFiles/cmt.dir/main.cpp.o: CMakeFiles/cmt.dir/flags.make
CMakeFiles/cmt.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gerrywalsh/Videos/Demo_editing/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/cmt.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cmt.dir/main.cpp.o -c /home/gerrywalsh/Videos/Demo_editing/main.cpp

CMakeFiles/cmt.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cmt.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gerrywalsh/Videos/Demo_editing/main.cpp > CMakeFiles/cmt.dir/main.cpp.i

CMakeFiles/cmt.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cmt.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gerrywalsh/Videos/Demo_editing/main.cpp -o CMakeFiles/cmt.dir/main.cpp.s

CMakeFiles/cmt.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/cmt.dir/main.cpp.o.requires

CMakeFiles/cmt.dir/main.cpp.o.provides: CMakeFiles/cmt.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/cmt.dir/build.make CMakeFiles/cmt.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/cmt.dir/main.cpp.o.provides

CMakeFiles/cmt.dir/main.cpp.o.provides.build: CMakeFiles/cmt.dir/main.cpp.o


CMakeFiles/cmt.dir/CMT.cpp.o: CMakeFiles/cmt.dir/flags.make
CMakeFiles/cmt.dir/CMT.cpp.o: CMT.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gerrywalsh/Videos/Demo_editing/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/cmt.dir/CMT.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cmt.dir/CMT.cpp.o -c /home/gerrywalsh/Videos/Demo_editing/CMT.cpp

CMakeFiles/cmt.dir/CMT.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cmt.dir/CMT.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gerrywalsh/Videos/Demo_editing/CMT.cpp > CMakeFiles/cmt.dir/CMT.cpp.i

CMakeFiles/cmt.dir/CMT.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cmt.dir/CMT.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gerrywalsh/Videos/Demo_editing/CMT.cpp -o CMakeFiles/cmt.dir/CMT.cpp.s

CMakeFiles/cmt.dir/CMT.cpp.o.requires:

.PHONY : CMakeFiles/cmt.dir/CMT.cpp.o.requires

CMakeFiles/cmt.dir/CMT.cpp.o.provides: CMakeFiles/cmt.dir/CMT.cpp.o.requires
	$(MAKE) -f CMakeFiles/cmt.dir/build.make CMakeFiles/cmt.dir/CMT.cpp.o.provides.build
.PHONY : CMakeFiles/cmt.dir/CMT.cpp.o.provides

CMakeFiles/cmt.dir/CMT.cpp.o.provides.build: CMakeFiles/cmt.dir/CMT.cpp.o


CMakeFiles/cmt.dir/Consensus.cpp.o: CMakeFiles/cmt.dir/flags.make
CMakeFiles/cmt.dir/Consensus.cpp.o: Consensus.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gerrywalsh/Videos/Demo_editing/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/cmt.dir/Consensus.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cmt.dir/Consensus.cpp.o -c /home/gerrywalsh/Videos/Demo_editing/Consensus.cpp

CMakeFiles/cmt.dir/Consensus.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cmt.dir/Consensus.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gerrywalsh/Videos/Demo_editing/Consensus.cpp > CMakeFiles/cmt.dir/Consensus.cpp.i

CMakeFiles/cmt.dir/Consensus.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cmt.dir/Consensus.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gerrywalsh/Videos/Demo_editing/Consensus.cpp -o CMakeFiles/cmt.dir/Consensus.cpp.s

CMakeFiles/cmt.dir/Consensus.cpp.o.requires:

.PHONY : CMakeFiles/cmt.dir/Consensus.cpp.o.requires

CMakeFiles/cmt.dir/Consensus.cpp.o.provides: CMakeFiles/cmt.dir/Consensus.cpp.o.requires
	$(MAKE) -f CMakeFiles/cmt.dir/build.make CMakeFiles/cmt.dir/Consensus.cpp.o.provides.build
.PHONY : CMakeFiles/cmt.dir/Consensus.cpp.o.provides

CMakeFiles/cmt.dir/Consensus.cpp.o.provides.build: CMakeFiles/cmt.dir/Consensus.cpp.o


CMakeFiles/cmt.dir/Fusion.cpp.o: CMakeFiles/cmt.dir/flags.make
CMakeFiles/cmt.dir/Fusion.cpp.o: Fusion.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gerrywalsh/Videos/Demo_editing/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/cmt.dir/Fusion.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cmt.dir/Fusion.cpp.o -c /home/gerrywalsh/Videos/Demo_editing/Fusion.cpp

CMakeFiles/cmt.dir/Fusion.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cmt.dir/Fusion.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gerrywalsh/Videos/Demo_editing/Fusion.cpp > CMakeFiles/cmt.dir/Fusion.cpp.i

CMakeFiles/cmt.dir/Fusion.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cmt.dir/Fusion.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gerrywalsh/Videos/Demo_editing/Fusion.cpp -o CMakeFiles/cmt.dir/Fusion.cpp.s

CMakeFiles/cmt.dir/Fusion.cpp.o.requires:

.PHONY : CMakeFiles/cmt.dir/Fusion.cpp.o.requires

CMakeFiles/cmt.dir/Fusion.cpp.o.provides: CMakeFiles/cmt.dir/Fusion.cpp.o.requires
	$(MAKE) -f CMakeFiles/cmt.dir/build.make CMakeFiles/cmt.dir/Fusion.cpp.o.provides.build
.PHONY : CMakeFiles/cmt.dir/Fusion.cpp.o.provides

CMakeFiles/cmt.dir/Fusion.cpp.o.provides.build: CMakeFiles/cmt.dir/Fusion.cpp.o


CMakeFiles/cmt.dir/Matcher.cpp.o: CMakeFiles/cmt.dir/flags.make
CMakeFiles/cmt.dir/Matcher.cpp.o: Matcher.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gerrywalsh/Videos/Demo_editing/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/cmt.dir/Matcher.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cmt.dir/Matcher.cpp.o -c /home/gerrywalsh/Videos/Demo_editing/Matcher.cpp

CMakeFiles/cmt.dir/Matcher.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cmt.dir/Matcher.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gerrywalsh/Videos/Demo_editing/Matcher.cpp > CMakeFiles/cmt.dir/Matcher.cpp.i

CMakeFiles/cmt.dir/Matcher.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cmt.dir/Matcher.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gerrywalsh/Videos/Demo_editing/Matcher.cpp -o CMakeFiles/cmt.dir/Matcher.cpp.s

CMakeFiles/cmt.dir/Matcher.cpp.o.requires:

.PHONY : CMakeFiles/cmt.dir/Matcher.cpp.o.requires

CMakeFiles/cmt.dir/Matcher.cpp.o.provides: CMakeFiles/cmt.dir/Matcher.cpp.o.requires
	$(MAKE) -f CMakeFiles/cmt.dir/build.make CMakeFiles/cmt.dir/Matcher.cpp.o.provides.build
.PHONY : CMakeFiles/cmt.dir/Matcher.cpp.o.provides

CMakeFiles/cmt.dir/Matcher.cpp.o.provides.build: CMakeFiles/cmt.dir/Matcher.cpp.o


CMakeFiles/cmt.dir/Tracker.cpp.o: CMakeFiles/cmt.dir/flags.make
CMakeFiles/cmt.dir/Tracker.cpp.o: Tracker.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gerrywalsh/Videos/Demo_editing/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/cmt.dir/Tracker.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cmt.dir/Tracker.cpp.o -c /home/gerrywalsh/Videos/Demo_editing/Tracker.cpp

CMakeFiles/cmt.dir/Tracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cmt.dir/Tracker.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gerrywalsh/Videos/Demo_editing/Tracker.cpp > CMakeFiles/cmt.dir/Tracker.cpp.i

CMakeFiles/cmt.dir/Tracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cmt.dir/Tracker.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gerrywalsh/Videos/Demo_editing/Tracker.cpp -o CMakeFiles/cmt.dir/Tracker.cpp.s

CMakeFiles/cmt.dir/Tracker.cpp.o.requires:

.PHONY : CMakeFiles/cmt.dir/Tracker.cpp.o.requires

CMakeFiles/cmt.dir/Tracker.cpp.o.provides: CMakeFiles/cmt.dir/Tracker.cpp.o.requires
	$(MAKE) -f CMakeFiles/cmt.dir/build.make CMakeFiles/cmt.dir/Tracker.cpp.o.provides.build
.PHONY : CMakeFiles/cmt.dir/Tracker.cpp.o.provides

CMakeFiles/cmt.dir/Tracker.cpp.o.provides.build: CMakeFiles/cmt.dir/Tracker.cpp.o


CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.o: CMakeFiles/cmt.dir/flags.make
CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.o: fastcluster/fastcluster.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gerrywalsh/Videos/Demo_editing/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.o -c /home/gerrywalsh/Videos/Demo_editing/fastcluster/fastcluster.cpp

CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gerrywalsh/Videos/Demo_editing/fastcluster/fastcluster.cpp > CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.i

CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gerrywalsh/Videos/Demo_editing/fastcluster/fastcluster.cpp -o CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.s

CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.o.requires:

.PHONY : CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.o.requires

CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.o.provides: CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.o.requires
	$(MAKE) -f CMakeFiles/cmt.dir/build.make CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.o.provides.build
.PHONY : CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.o.provides

CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.o.provides.build: CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.o


# Object files for target cmt
cmt_OBJECTS = \
"CMakeFiles/cmt.dir/common.cpp.o" \
"CMakeFiles/cmt.dir/gui.cpp.o" \
"CMakeFiles/cmt.dir/main.cpp.o" \
"CMakeFiles/cmt.dir/CMT.cpp.o" \
"CMakeFiles/cmt.dir/Consensus.cpp.o" \
"CMakeFiles/cmt.dir/Fusion.cpp.o" \
"CMakeFiles/cmt.dir/Matcher.cpp.o" \
"CMakeFiles/cmt.dir/Tracker.cpp.o" \
"CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.o"

# External object files for target cmt
cmt_EXTERNAL_OBJECTS =

cmt: CMakeFiles/cmt.dir/common.cpp.o
cmt: CMakeFiles/cmt.dir/gui.cpp.o
cmt: CMakeFiles/cmt.dir/main.cpp.o
cmt: CMakeFiles/cmt.dir/CMT.cpp.o
cmt: CMakeFiles/cmt.dir/Consensus.cpp.o
cmt: CMakeFiles/cmt.dir/Fusion.cpp.o
cmt: CMakeFiles/cmt.dir/Matcher.cpp.o
cmt: CMakeFiles/cmt.dir/Tracker.cpp.o
cmt: CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.o
cmt: CMakeFiles/cmt.dir/build.make
cmt: /usr/local/lib/libopencv_cudabgsegm.so.3.4.0
cmt: /usr/local/lib/libopencv_cudaobjdetect.so.3.4.0
cmt: /usr/local/lib/libopencv_cudastereo.so.3.4.0
cmt: /usr/local/lib/libopencv_stitching.so.3.4.0
cmt: /usr/local/lib/libopencv_superres.so.3.4.0
cmt: /usr/local/lib/libopencv_videostab.so.3.4.0
cmt: /usr/local/lib/libopencv_aruco.so.3.4.0
cmt: /usr/local/lib/libopencv_bgsegm.so.3.4.0
cmt: /usr/local/lib/libopencv_bioinspired.so.3.4.0
cmt: /usr/local/lib/libopencv_ccalib.so.3.4.0
cmt: /usr/local/lib/libopencv_dpm.so.3.4.0
cmt: /usr/local/lib/libopencv_face.so.3.4.0
cmt: /usr/local/lib/libopencv_freetype.so.3.4.0
cmt: /usr/local/lib/libopencv_fuzzy.so.3.4.0
cmt: /usr/local/lib/libopencv_hdf.so.3.4.0
cmt: /usr/local/lib/libopencv_img_hash.so.3.4.0
cmt: /usr/local/lib/libopencv_line_descriptor.so.3.4.0
cmt: /usr/local/lib/libopencv_optflow.so.3.4.0
cmt: /usr/local/lib/libopencv_reg.so.3.4.0
cmt: /usr/local/lib/libopencv_rgbd.so.3.4.0
cmt: /usr/local/lib/libopencv_saliency.so.3.4.0
cmt: /usr/local/lib/libopencv_stereo.so.3.4.0
cmt: /usr/local/lib/libopencv_structured_light.so.3.4.0
cmt: /usr/local/lib/libopencv_surface_matching.so.3.4.0
cmt: /usr/local/lib/libopencv_tracking.so.3.4.0
cmt: /usr/local/lib/libopencv_xfeatures2d.so.3.4.0
cmt: /usr/local/lib/libopencv_ximgproc.so.3.4.0
cmt: /usr/local/lib/libopencv_xobjdetect.so.3.4.0
cmt: /usr/local/lib/libopencv_xphoto.so.3.4.0
cmt: /usr/local/lib/libopencv_cudafeatures2d.so.3.4.0
cmt: /usr/local/lib/libopencv_shape.so.3.4.0
cmt: /usr/local/lib/libopencv_cudacodec.so.3.4.0
cmt: /usr/local/lib/libopencv_cudaoptflow.so.3.4.0
cmt: /usr/local/lib/libopencv_cudalegacy.so.3.4.0
cmt: /usr/local/lib/libopencv_cudawarping.so.3.4.0
cmt: /usr/local/lib/libopencv_photo.so.3.4.0
cmt: /usr/local/lib/libopencv_cudaimgproc.so.3.4.0
cmt: /usr/local/lib/libopencv_cudafilters.so.3.4.0
cmt: /usr/local/lib/libopencv_cudaarithm.so.3.4.0
cmt: /usr/local/lib/libopencv_datasets.so.3.4.0
cmt: /usr/local/lib/libopencv_plot.so.3.4.0
cmt: /usr/local/lib/libopencv_text.so.3.4.0
cmt: /usr/local/lib/libopencv_dnn.so.3.4.0
cmt: /usr/local/lib/libopencv_ml.so.3.4.0
cmt: /usr/local/lib/libopencv_video.so.3.4.0
cmt: /usr/local/lib/libopencv_calib3d.so.3.4.0
cmt: /usr/local/lib/libopencv_features2d.so.3.4.0
cmt: /usr/local/lib/libopencv_highgui.so.3.4.0
cmt: /usr/local/lib/libopencv_videoio.so.3.4.0
cmt: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.0
cmt: /usr/local/lib/libopencv_flann.so.3.4.0
cmt: /usr/local/lib/libopencv_imgcodecs.so.3.4.0
cmt: /usr/local/lib/libopencv_objdetect.so.3.4.0
cmt: /usr/local/lib/libopencv_imgproc.so.3.4.0
cmt: /usr/local/lib/libopencv_core.so.3.4.0
cmt: /usr/local/lib/libopencv_cudev.so.3.4.0
cmt: CMakeFiles/cmt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gerrywalsh/Videos/Demo_editing/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX executable cmt"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cmt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cmt.dir/build: cmt

.PHONY : CMakeFiles/cmt.dir/build

CMakeFiles/cmt.dir/requires: CMakeFiles/cmt.dir/common.cpp.o.requires
CMakeFiles/cmt.dir/requires: CMakeFiles/cmt.dir/gui.cpp.o.requires
CMakeFiles/cmt.dir/requires: CMakeFiles/cmt.dir/main.cpp.o.requires
CMakeFiles/cmt.dir/requires: CMakeFiles/cmt.dir/CMT.cpp.o.requires
CMakeFiles/cmt.dir/requires: CMakeFiles/cmt.dir/Consensus.cpp.o.requires
CMakeFiles/cmt.dir/requires: CMakeFiles/cmt.dir/Fusion.cpp.o.requires
CMakeFiles/cmt.dir/requires: CMakeFiles/cmt.dir/Matcher.cpp.o.requires
CMakeFiles/cmt.dir/requires: CMakeFiles/cmt.dir/Tracker.cpp.o.requires
CMakeFiles/cmt.dir/requires: CMakeFiles/cmt.dir/fastcluster/fastcluster.cpp.o.requires

.PHONY : CMakeFiles/cmt.dir/requires

CMakeFiles/cmt.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cmt.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cmt.dir/clean

CMakeFiles/cmt.dir/depend:
	cd /home/gerrywalsh/Videos/Demo_editing && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gerrywalsh/Videos/Demo_editing /home/gerrywalsh/Videos/Demo_editing /home/gerrywalsh/Videos/Demo_editing /home/gerrywalsh/Videos/Demo_editing /home/gerrywalsh/Videos/Demo_editing/CMakeFiles/cmt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cmt.dir/depend
