# AdamSebEd

CS-433 — Project 1

Predicting coronary heart disease (MICHD) risk from BRFSS survey data.

Team: Edouard Fousson, Adam, Seb Deadline: Thursday, October 29 2026, 16:00 — submission at http://mlcourse.epfl.ch

Getting started

Everything below is a one-time setup. Ping the group chat if anything fails.

1. Install the tools

git --version
conda --version
code --version

Install what's missing:

2. Configure git

Once per machine:

git config --global user.name "Your Name"
git config --global user.email "your.github.email@example.com"

Use the email tied to your GitHub account, otherwise your commits won't be linked to your profile.

3. Clone both repos

Side by side, never one inside the other:

cd C:\Users\YOU\Documents
git clone https://github.com/epfml/ML_course.git  
git clone https://github.com/EdouardFsn/AdamSebEd.git

The course repo is needed for the grading tests.

4. Get the data

The data/ folder is not versioned — the CSVs are too large for git, so everyone downloads their own copy.

Create an AIcrowd account with your @epfl.ch address.
Join the challenge: https://www.aicrowd.com/challenges/epfl-machine-learning-project-1
Download x_train.csv, y_train.csv and x_test.csv.
Create a data/ folder at the repo root and put the three files in it.
Do not rename them — load_csv_data expects these exact names.

Sanity check: git status should show no .csv files.

5. Create the grading environment


cd C:\Users\YOURNAME\ML_course\projects\project1\grading_tests   (or your own personal path to the ML_course on your own machine)
conda env create --file=environment.yml --name=project1-grading
conda activate project1-grading

In VS Code: open the AdamSebEd folder, then Ctrl+Shift+P → Python: Select Interpreter → pick project1-grading.

That environment ships scikit-learn and pandas for running the tests only. We are not allowed to use them in our code.

6. Enable auto-formatting

Ctrl+Shift+P → Preferences: Open User Settings (JSON), then add this block before the closing brace (mind the comma on the line above):

"[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true
}

Please do this. It only pays off if all three of us do — otherwise git flags conflicts on lines nobody actually changed.

Running the tests
cd C:\Users\YOURNAME\ML_course\projects\project1\grading_tests    
conda activate project1-grading
pytest --github_link C:\Users\YOURNAME\Documents\AdamSebEd .

To target a single function:

pytest --github_link C:\Users\YOURNAME\Documents\AdamSebEd . -k least_squares

Two failures are expected and harmless during development:

value tests failing with NotImplementedError — that function isn't written yet
test_github_link_format — we pass a local path instead of a URL

The tests may be updated by the TAs, so pull the course repo now and then (we will also get the helpers.py later i think):

cd C:\Users\YOURNAME\ML_course  (or still your own path)
git pull



Daily workflow

Start of session:

git pull

End of session:

git add .
git commit -m "implement ridge_regression"
git push

Commit messages in English, imperative mood, short.


Repo structure
AdamSebEd/
├── README.md              this file
├── implementations.py     the six required functions
├── run.py                 reproduces our best AIcrowd submission
├── helpers.py             data loading and submission writing
├── data/                  not versioned — see step 4
└── .gitignore

README.md, implementations.py and run.py must stay at the root — the grading tests check for them there.

Required functions

All six go in implementations.py, with these exact signatures:

Function	Signature
mean_squared_error_gd	(y, tx, initial_w, max_iters, gamma)
mean_squared_error_sgd	(y, tx, initial_w, max_iters, gamma)
least_squares	(y, tx)
ridge_regression	(y, tx, lambda_)
logistic_regression	(y, tx, initial_w, max_iters, gamma)
reg_logistic_regression	(y, tx, lambda_, initial_w, max_iters, gamma)



Conventions enforced by the tests:

return (w, loss) — only the last w, not the whole history
loss must be a scalar (loss.ndim == 0), not a size-1 array
w must have shape (D,), never (D, 1)
MSE carries a factor 0.5
ridge and regularized logistic return the loss without the penalty term
SGD uses mini-batch size 1
logistic regression expects y ∈ {0, 1}, not {-1, 1}
least_squares may use anything from numpy.linalg except lstsq
max_iters=0 must work and return initial_w with its loss
every function needs a docstring
the word "TODO" must not appear in any .py file
Project constraints
Allowed: Python standard library, NumPy. matplotlib/seaborn for plots only.
Not allowed: pandas, scikit-learn, PyTorch, TensorFlow, any external library or dataset.
The repo must stay public.
AIcrowd allows 5 submissions per day. Always validate locally too — the leaderboard score is not a substitute for cross-validation.
Project 1 does not count toward the final grade; it's preparation for Project 2 (30%).








