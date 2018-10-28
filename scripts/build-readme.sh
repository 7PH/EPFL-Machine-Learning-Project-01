#!/usr/bin/env bash

OUTFILE="README.md";
TODO=$(grep -in --color "@TODO" ./src/*.py);

CONTENT="
# How to use (UNIX)

First, you need to clone the repository.

Then, you can download the CSV files using

\`\`\`bash
npm run setup
\`\`\`

To ensure your setup is ok, you can run

\`\`\`bash
npm test
\`\`\`

Then, before a commit, you need to regenerate the README file using
\`\`\`bash
npm run build
\`\`\`

To run the python linter, use
\`\`\`bash
npm run lint
\`\`\`

# @TODO

The content below is automatically generated
\`\`\`text
$TODO
\`\`\`";


echo "$CONTENT" > "$OUTFILE";

cat "$OUTFILE";