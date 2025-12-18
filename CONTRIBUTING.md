Contributing
The following is a set of guidelines for contributing to graphite-maps
All code must be testable and unit tested.

Commits
We strive to keep a consistent and clean git history and all contributions should adhere to the following:

All tests should pass on all commits(*)
A commit should do one atomic change on the repository
The commit message should be descriptive.
We expect commit messages to follow the style described here. Also, focus on making clear the reasons why you made the change in the first place—the way things worked before the change (and what was wrong with that), the way they work now, and why you decided to solve it the way you did. A commit body is required for anything except very small changes.

(*) Tip for making sure all tests passes, try out --exec while rebasing. You can then have all tests run per commit in a single command.

Pull Request Process
Work on your own fork of the main repo
Push your commits and make a draft pull request using the pull request template.
Check that your pull request passes all tests.
When all tests have passed and your are happy with your changes, change your pull request to "ready for review" and ask for a code review.
