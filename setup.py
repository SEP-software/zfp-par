from skbuild import setup
cmake_args = [
    '-DINSTALL_INCLUDE_FILES=NO',
    '-DBUILD_ZFPY=YES',
    '-DZFP_WITH_OPENMP=yes'
]

setup(
    name="zfpy",
    zip_safe=False,
    packages=find_packages(),
    include_package_data=True,
    cmake_args=cmake_args,
    version="1.0.2",
    author="Peter Lindstrom, Danielle Asher",
    author_email="zfp@llnl.gov",
    url="https://zfp.llnl.gov",
    description="zfp compression in Python",
)
