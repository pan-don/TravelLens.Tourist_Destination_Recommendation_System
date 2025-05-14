<<<<<<< Updated upstream
from scripts.execute_notebook import run_notebook


if __name__ == "__main__":    
    run_notebook('scripts/notebook/training_model.ipynb')
=======
from app.main.app import app
# from scripts.execute_notebook import run_notebook

if __name__ == '__main__':
    app.run(debug=True)
    # run_notebook('scripts/notebook/training_model.ipynb')  
>>>>>>> Stashed changes
