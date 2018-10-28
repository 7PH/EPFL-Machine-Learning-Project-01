
# How to use (UNIX)

First, you need to clone the repository.

Then, you can download the CSV files using

```bash
npm run setup
# or
sh scripts/setup.sh
```

To run the unit tests and ensure your environment is correctly setup, you must run

```bash
npm test
# or
sh scripts/test-karma.sh
sh scripts/test-unit.sh
```

Before a commit, you need to regenerate the README file using
```bash
npm run build
# or
sh scripts/build.sh
```

To start the `run.sh` file, use
```bash
npm start
# or
python -m src.run
```

# @TODO

The content below is automatically generated
```text
./src/augmentation.py:7:    @TODO document
./src/augmentation.py:20:    @TODO document
./src/augmentation.py:33:    @TODO remove? document?
./src/augmentation.py:47:    @TODO document
./src/implementations.py:66:    @TODO document
./src/kfold.py:16:    @TODO document
./src/kfold.py:44:    @TODO document? move?
./src/kfold.py:70:    @TODO document
./src/kfold.py:96:    @TODO move to augmentation?
./src/kfold.py:122:    # @TODO check that my modification did not fuck up everything
./src/kfold.py:134:    @TODO refactor & document
./src/kfold.py:153:    @TODO document
./src/kfold.py:170:    @TODO document
./src/losses.py:13:    @TODO document
./src/losses.py:25:    @TODO refactor (important)
./src/losses.py:38:    @TODO refactor (important)
./src/run.py:11:    @TODO document
./src/run.py:43:    @TODO document
./src/utils.py:6:    @TODO move this (logistic helper)
./src/utils.py:19:    @TODO move this (logistic helper)
```
