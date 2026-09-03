# Source this (or let use_zerocopy_pyrealsense.sh / activate hook do it) so
# pyrealsense2 loads the CUDA+ZEROCOPY librealsense2 from /usr/local instead of
# ROS Jazzy's copy.
#
#   source scripts/rs_zerocopy_env.sh

export LD_LIBRARY_PATH=/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=/usr/local/bin${PATH:+:$PATH}
