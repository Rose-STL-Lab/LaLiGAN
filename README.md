# Rose-STL Lab Code Quality Framework

This repository is the starting point from which all repositories should be prepared for code release. The following are the different code quality checks in this repository that should be ported to your own repo.

## 1. Automated Github Actions

These are [github actions](https://github.com/features/actions) configured to perform the checks when triggered. The configurations are located in .github/workflows

| Check     | Triggers  | Description   |
| ---       | ---       | ---           |
| Files     | push          |  Edit files in ```.github/check_run_scripts.sh```. The overall idea is to make sure the repo contains runnable enviornment setup, data processing, and experiment scripts.           |
| Lint      | push          |  Check lint in local enviornment first. This can be achieved by downloading the ```Black Formatter``` and ```Flake8``` extensions in VSCode.            |

## 2. Custom checks via Github Workflows

The are custom script based checks triggered using github actions. The scripts are located in check_scripts. Each script has a corresponding triggering workflow in .github/workflows

## 3. Human code reviews

The main branch will not be allowed direct commits. This is managed by the protected branch setting in the settings tab of the repo. The repo owner needs to set this up by creating the same rules. Any development shall be done on branches and code should be merged to main via a pull request before release. The steps to follow for the same are:

1. Raise a pull request and make sure all automated checks pass.
2. Add 2 eligible reviewers to the pull request.
3. The reviewers review the pull request and add comments for any modifications as might be needed.
4. This review cycle continues till the code reaches satisfactory quality.
5. Once ready the code can be merged by an eligible person to the main branch and is ready for release.

## 4. Reproducibility using Docker

--

## Miscellaneous Trouble shooting

### Linting Issues

In VSCode, Open User Settings (JSON) command in the Command Palette (Ctrl+Shift+P). You can edit max line length and ignore list as follows. Remember to update the list in your ```pylinter.yml``` as well.

```
{
    "window.zoomLevel": 1,
    "editor.inlineSuggest.enabled": true,
    "[python]": {
        "editor.formatOnType": true
    },
    "notebook.output.scrolling": true,
    "flake8.args": [
        "--max-line-length=150",
        "--ignore=E402,E731,F541,W291,E122,E127,F401,E266,E241,C901,E741,W293,F811,W503,E203,F403,F405"
    ],
    "git.enableSmartCommit": true,
    "git.confirmSync": false
}
```
