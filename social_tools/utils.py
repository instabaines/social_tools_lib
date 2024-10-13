import concurrent.futures
from typing import Callable, List, Any

def parallelize(func: Callable, inputs: List[Any], max_workers: int = 5) -> List[Any]:
    """
    function to parallelize the execution of any given function.
    
    Args:
        func (Callable): The function to execute in parallel.
        inputs (List[Any]): A list of inputs to process in parallel.
        max_workers (int): Maximum number of worker threads or processes to use.
        
    Returns:
        List[Any]: List of results from each function execution.
    """
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each function call to the executor
        future_to_input = {executor.submit(func, inp): inp for inp in inputs}
        
        # Collect the results as they complete
        for future in concurrent.futures.as_completed(future_to_input):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'Generated an exception: {exc}')
    
    return results
