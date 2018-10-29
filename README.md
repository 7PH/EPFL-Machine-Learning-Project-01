
# How to use (UNIX)

First, you need to clone the repository.

```bash
git clone https://github.com/7PH/EPFL-Machine-Learning-Project-01.git;
cd EPFL-Machine-Learning-Project-01;
```

Then, you can download the CSV files using

```bash
npm run setup
# or
sh scripts/setup.sh
```

**Do not forget to activate the python virtual environment you want to use at this point**

To run the unit tests and ensure your environment is correctly setup, you must run

```bash
npm test
# or
sh scripts/test-karma.sh
sh scripts/test-unit.sh
```

To start the `run.sh` file, use
```bash
npm start
# or
python3 run.py
```

# Contributing

Before a commit, you need to regenerate the README file using
```bash
npm run build
# or
sh scripts/build.sh
```

# @TODO

The content below is automatically generated
```text
./src/augmentation.py:9:    @TODO document
./src/augmentation.py:22:    @TODO document
./src/augmentation.py:35:    @TODO remove? document?
./src/augmentation.py:49:    @TODO document
./src/augmentation.py:65:    @TODO document
./src/augmentation.py:92:    @TODO document? move? => yes please move :) 
./src/augmentation.py:118:    @TODO document
./src/implementations.py:62:    @TODO document
./src/kfold.py:11:    @TODO move to augmentation?
./src/kfold.py:37:    # @TODO check that my modification did not fuck up everything
./src/kfold.py:45:    @TODO refactor & document
./src/kfold.py:64:    @TODO document
./src/kfold.py:81:    @TODO document
./src/run_best_model.py:11:    @TODO document
./src/run_best_model.py:44:    @TODO document
./src/utils.py:6:    @TODO move this (logistic helper)
./src/utils.py:19:    @TODO move this (logistic helper)
```
