function noTypes(markdown) {
  if (/## Type of change/.test(markdown) && /- \[x\]/i.test(markdown)) {
    return false;
  }
  return true;
}

function noDescription(markdown) {
  return (
    /## Description/.test(markdown) === false ||
    /## Description\s*\n\s*## \w+/.test(markdown) ||
    /## Description\s*\n\s*$/.test(markdown)
  );
}

module.exports = async ({ github, context }) => {
  const pr = context.payload.pull_request;

  if (pr.labels.length > 0) {
    // Skip if the PR is already labeled (typically created by a deps-bot.)
    return;
  }

  const body = pr.body === null ? '' : pr.body.trim();
  const markdown = body.replace(/<!--[\s\S]*?-->/g, '');

  if (body === '' || noTypes(markdown) || noDescription(markdown)) {
    await github.rest.pulls.update({
      ...context.repo,
      pull_number: pr.number,
      state: 'closed'
    });

    await github.rest.issues.createComment({
      ...context.repo,
      issue_number: pr.number,
      body: "Oops, it seems you've submitted an invalid pull request. No worries, we'll close it for you."
    });
  }
};
