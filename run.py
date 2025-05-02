import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(path):
    with open(path) as f:
        nb = nbformat.read(path, as_version=4)
        
    ep = ExecutePreprocessor(timeout=500, kernel_name='python3')
    ep.preprocess(nb,{'metadata': {'path': './'}})


if __name__ == "__main__":
    run_notebook('scripts/notebook/ContentBasedFiltering.ipynb')