from skbuild import setup
cmake_args = [
    '-DINSTALL_INCLUDE_FILES=NO'
    "-DBUILD_ZFPY=YES"
]
setup(
        name="zfpy",
            cmake_args=cmake_args,
    version="1.0.2",
    author="Peter Lindstrom, Danielle Asher",
    author_email="zfp@llnl.gov",
    url="https://zfp.llnl.gov",
    description="zfp compression in Python",
    long_description="zfp is a compressed format for representing multidimensional floating-point and integer arrays. zfp provides compressed-array classes that support high throughput read and write random access to individual array elements. zfp also supports serial and parallel compression of whole arrays using both lossless and lossy compression with error tolerances. zfp is primarily written in C and C++ but also includes Python and Fortran bindings.",
)
