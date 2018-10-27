#!/usr/bin/env bash

OUTFILE="README.md";
TODO=$(grep -in --color "@TODO" ./*.py);

CONTENT="
# Dev
First, you need to clone the repository.

The CSV files test.csv and train.csv should be present in the data folder and are not included in this repository.

To ensure your setup is ok you can run

\`\`\`bash
npm test
\`\`\`

Then, before a commit, you need to regenerate the README file, use
\`\`\`bash
npm run build
\`\`\`

# @TODO

The content below is automatically generated
\`\`\`text
$TODO
\`\`\`";


echo "$CONTENT" > "$OUTFILE";

cat "$OUTFILE";