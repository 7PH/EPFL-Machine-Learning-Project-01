#!/usr/bin/env bash

OUTFILE="README.md";
TODO=$(grep -in --color "@TODO" ./*.py);

CONTENT="
# Dev
To install, clone the repo.
To regenerate the README file, use
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