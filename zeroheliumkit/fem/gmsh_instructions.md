# How to build Gmsh with p4est and Hxt

Regular `pip install gmsh` does NOT have p4est or Hxt. You have to build
Gmsh yourself from source to get them.

## 1. Install tools

```bash
brew install autoconf automake libtool open-mpi cmake fltk opencascade gmp
```

## 2. Build p4est

```bash
cd ~
git clone --recursive https://github.com/cburstedde/p4est.git
cd p4est
./bootstrap
mkdir -p build && cd build
../configure --enable-mpi --disable-shared CC=mpicc CXX=mpicxx --prefix=$HOME/local/p4est
make -j$(sysctl -n hw.ncpu) install
```

## 3. Build Gmsh

```bash
cd ~
git clone https://gitlab.onelab.info/gmsh/gmsh.git
cd gmsh
mkdir build && cd build
cmake -DENABLE_P4EST=1 -DENABLE_HXT=1 -DENABLE_MPI=ON -DENABLE_OPENMP=1 \
      -DENABLE_BUILD_DYNAMIC=1 \
      -DCMAKE_PREFIX_PATH="$HOME/local/p4est;/opt/homebrew" \
      -DCMAKE_C_COMPILER=$(brew --prefix open-mpi)/bin/mpicc \
      -DCMAKE_CXX_COMPILER=$(brew --prefix open-mpi)/bin/mpicxx ..
make -j$(sysctl -n hw.ncpu)
```

Your new gmsh binary is now at `~/gmsh/build/gmsh`.

## 4. Check it worked

```bash
~/gmsh/build/gmsh -info | grep -i p4est
~/gmsh/build/gmsh -info | grep -i hxt
```

Both should print something. If empty, it didn't build in.

## 5. One extra fix needed

If you plan to use `-bgm`/`-size_field` (the adaptive meshing pipeline),
you need one code patch or it will crash. In
`src/mesh/automaticMeshSizeField.cpp`, find `sc_MPI_Init(&argc, &argv);`
near the top of `forestCreate()` and wrap it like this:

```cpp
int already_init = 0;
MPI_Initialized(&already_init);
if (!already_init) {
    sc_MPI_Init(&argc, &argv);
}
```

Then rebuild:
```bash
cd ~/gmsh/build
make -j$(sysctl -n hw.ncpu)
```

## 6. Make sure Python uses this build, not pip's

```bash
python3 -c "import gmsh; print(gmsh.__file__)"
```

If it doesn't point to your new build, run:
```bash
pip uninstall gmsh
```