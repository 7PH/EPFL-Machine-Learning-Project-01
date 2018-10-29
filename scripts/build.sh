#!/usr/bin/env bash

OUTFILE="README.md";
TODO=$(grep -in --color "@TODO" ./src/*.py);

CONTENT="
# How to use (UNIX)

First, you need to clone the repository.

\`\`\`bash
git clone https://github.com/7PH/EPFL-Machine-Learning-Project-01.git;
cd EPFL-Machine-Learning-Project-01;
\`\`\`

Then, you can download the CSV files using

\`\`\`bash
npm run setup
# or
sh scripts/setup.sh
\`\`\`

**Do not forget to activate the python virtual environment you want to use at this point**

To run the unit tests and ensure your environment is correctly setup, you must run

\`\`\`bash
npm test
# or
sh scripts/test-karma.sh
sh scripts/test-unit.sh
\`\`\`

To start the \`run.sh\` file, use
\`\`\`bash
npm start
# or
python3 run.py
\`\`\`

# Contributing

Before a commit, you need to regenerate the README file using
\`\`\`bash
npm run build
# or
sh scripts/build.sh
\`\`\`

# @TODO

The content below is automatically generated
\`\`\`text
$TODO
\`\`\`";


echo "$CONTENT" > "$OUTFILE";

cat "$OUTFILE";