echo "Starting installation script..."

if [ -d build ]; then
    rm -rf build
fi

mkdir build
cp 3rdparty/tvm/cmake/config.cmake build
cd build


echo "Configuring TVM build with LLVM and CUDA paths..."
echo "set(USE_LLVM llvm-config-16)" >> config.cmake && echo "set(USE_ROCM /opt/rocm)" >> config.cmake

echo "Running CMake for TileLang..."
cmake ..
if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed."
    exit 1
fi

echo "Building TileLang with make..."
make -j
if [ $? -ne 0 ]; then
    echo "Error: TileLang build failed."
    exit 1
else
    echo "TileLang build completed successfully."
fi

cd ..


# Define the lines to be added
TILELANG_PATH="$(pwd)"
echo "Configuring environment variables for TVM..."
echo "export PYTHONPATH=${TILELANG_PATH}:\$PYTHONPATH" >> ~/.bashrc
TVM_HOME_ENV="export TVM_HOME=${TILELANG_PATH}/3rdparty/tvm"
TILELANG_PYPATH_ENV="export PYTHONPATH=\$TVM_HOME/python:${TILELANG_PATH}:\$PYTHONPATH"

# Check and add the first line if not already present
if ! grep -qxF "$TVM_HOME_ENV" ~/.bashrc; then
    echo "$TVM_HOME_ENV" >> ~/.bashrc
    echo "Added TVM_HOME to ~/.bashrc"
else
    echo "TVM_HOME is already set in ~/.bashrc"
fi

# Check and add the second line if not already present
if ! grep -qxF "$TILELANG_PYPATH_ENV" ~/.bashrc; then
    echo "$TILELANG_PYPATH_ENV" >> ~/.bashrc
    echo "Added PYTHONPATH to ~/.bashrc"
else
    echo "PYTHONPATH is already set in ~/.bashrc"
fi

# Reload ~/.bashrc to apply the changes
source ~/.bashrc

echo "Installation script completed successfully."
