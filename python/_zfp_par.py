import numpy as np
from multiprocess import RawArray, Pool, cpu_count
import math
from zfpy._zfpy import(block_compression, zfp_chunkit,
                       compress_numpy_portion, 
                       decompress_numpy_portion)

class zfp_p:
    def __init__(self, shape: tuple, dtype: np.dtype, est_compression_rate: float = 3,
                 method="BEST_CACHE", block_size: float = -1, nparts: int = -1):
        """Initialize the zfp compressed array.

        Args:
            shape (tuple): Tuple containing shape of the array.
            dtype (np.dtype): Numpy type of storage.
            est_compression_rate (float): Estimated compression ratio.
            method (str): Method to break block (BEST_CACHE or MAKE_EQUAL).
            block_size (float): Approximate size for each block (for method 1).
            nparts (int): Number of parts to break array into (for method 2).
        """
        # Mapping of NumPy dtype to typecode and size
        
       
        np_type_to_code = {
            'float32': 'f',
            'float64': 'd',
            'int32': 'i',
            'int64': 'q',
        }
        np_type_to_size = {
            'float32': 4,
            'float64': 8,
            'int32': 4,
            'int64': 8,
        }

        # Validate dtype
        if dtype not in np_type_to_code:
            raise ValueError(f"Unsupported NumPy dtype: {dtype}")
        
        # Validate shape dimensions
        if len(shape) > 4:
            raise ValueError("Only support up to 4 dimensions")

        # Calculate number of blocks and total elements
        nblocks = 1
        n123 = 1
        for x in shape: 
            nblocks *= int((x + 3) / 4)
            n123 *= x
            
        # Calculate chunks per block
        if block_size != -1:
            compress_block_size = math.pow(2, len(shape)) * np_type_to_size[dtype] / est_compression_rate
            chunks_per_block = block_size / compress_block_size
        elif nparts != -1:
            chunks_per_block = nblocks / nparts
        else:
            raise ValueError("Either block_size or nparts must be specified")

        # Create a RawArray
        total_size = n123
        self._raw_arr = RawArray(np_type_to_code[dtype], total_size)

        # Create NumPy array view
        self._np_array = np.frombuffer(self._raw_arr, dtype=dtype).reshape(shape)
        
        # Process compression blocks and chunks
        self._chunks = block_compression(self._np_array, chunks_per_block, method)
        self._chunkit = zfp_chunkit(len(shape), list(shape), self._chunks, self._np_array.dtype)

    def get_raw_array(self):
        """Return raw array representation"""
        return self._raw_arr

    def get_numpy_array(self):
        """Return numpy array representation"""
        return self._np_array
    
    def _init_pool(self, nthreads):
        # Initialize the multiprocessing pool
        if nthreads == -1:
            pool = Pool(initializer=self._init_shared_array, initargs=(self._raw_arr,))
        else:
            pool = Pool(processes=nthreads, initializer=self._init_shared_array, initargs=(self._raw_arr,))
        return pool

    @staticmethod
    def _pool_initializer(shared_arr_, chunkit_):
        global shared_arr
        global chunkit
        shared_arr = shared_arr_
        chunkit = chunkit_
    
    @staticmethod
    def _compress_chunk(ichunk, tolerance, rate, precision):
        # Wrap the shared RawArray in a NumPy array
        np_arr = np.frombuffer(shared_arr, dtype=chunkit.get_dtype()).reshape(chunkit.get_shape())
        # Compress the chunk
        return compress_numpy_portion(shared_arr, chunkit, ichunk, tolerance, rate, precision)


    def compress(self,nthreads=-1,tolerance = -1,rate = -1,precision = -1):
        """Compress 
            nthreads   - Number of threads (defaults to all)
            tolerance  - Error tollerance
            rate       - Fixed byte rate
            precission - Precision to keep
        """
        
        if nthreads==-1:
            nthreads=cpu_count()
        
        """Parallelized compress method."""
        self._compress_data = []

        # Initialize multiprocessing pool
        pool = Pool(processes=nthreads, initializer=self._pool_initializer, 
                    initargs=(self._raw_arr, self._chunkit))

        # Create tasks for the pool
        tasks = [(ichunk, tolerance, rate, precision) for ichunk in range(self._chunkit.get_nchunks())]

        # Execute the tasks in parallel
        self._compress_data = pool.starmap(self._compress_chunk, tasks)

        # Close the pool and wait for tasks to complete
        pool.close()
        pool.join()
    
    @staticmethod
    def _decompress_chunk(ichunk):
        # Wrap the shared RawArray in a NumPy array
        np_arr = np.frombuffer(shared_arr, dtype=chunkit.get_dtype()).reshape(chunkit.get_shape())
        # Decompress the chunk
        decompress_numpy_portion(compress_data[ichunk], np_arr, chunkit, ichunk)

    def decompress(self,nthreads=-1):
        """Compress array
            nthreads - Number of threads to use (defaults to all)
        """
        
        if nthreads==-1:
            nthreads=cpu_count()
            
        global compress_data
        compress_data = self._compress_data
        # Initialize multiprocessing pool
        pool = Pool(processes=nthreads, initializer=self._pool_initializer, 
                    initargs=(self._raw_arr, self._chunkit))
        # Create tasks for the pool
        tasks = [ichunk for ichunk in range(self._chunkit.get_nchunks())]

        # Execute the tasks in parallel
        pool.map(self._decompress_chunk, tasks)

        # Close the pool and wait for tasks to complete
        pool.close()
        pool.join()

def write_json_header(filename, dimensions, block_splits, compressed_files):
    """
    Writes a JSON header to a file.

    :param filename: The file to write the JSON header to.
    :param dimensions: A tuple representing the dimensions of the cube (1D to 4D).
    :param block_splits: A list of lists indicating how the cube is broken up along each axis.
    :param compressed_files: Either a single filename or a list of filenames for the compressed data.
    """
    header = {
        "dimensions": dimensions,
        "block_splits": block_splits,
        "compressed_files": compressed_files
    }

    with open(filename, 'w') as file:
        json.dump(header, file, indent=4)


def read_json_header(filename):
    """
    Reads a JSON header from a file.

    :param filename: The file to read the JSON header from.
    :return: A dictionary containing the header information.
    """
    with open(filename, 'r') as file:
        header = json.load(file)
    
    return header

