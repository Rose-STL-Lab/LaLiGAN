# Rose-STL Lab Code Quality Framework

This repository is the starting point from which all repositories should be prepared for code release. The following are the different code quality checks in this repository that should be ported to your own repo.

## 1. Automated Github Actions

These are [github actions](https://github.com/features/actions) configured to perform the checks when triggered. The configurations are located in .github/workflows

| Check     | Triggers  | Description   |
| ---       | ---       | ---           |
|           |           |               |
|           |           |               |

## 2. Custom checks via Github Workflows

The are custom script based checks triggered using github actions. The scripts are located in check_scripts. Each script has a corresponding triggering workflow in .github/workflows

## 3. Human code reviews

The main branch will not be allowed direct commits. This is managed by the protected branch setting in the settings tab of the repo. The setting can be found under Settings > Code and automation > Branch protection rules. Make sure all components of the rule are the same.


The repo owner needs to set this up by creating the same rules. Any development shall be done on branches and code should be merged to main via a pull request before release. The steps to follow for the same are:

1. Raise a pull request and make sure all automated checks pass.
2. Add 2 eligible reviewers (At least 1 with merge access and 1 with read acces) to the pull request.
3. The reviewers review the pull request and add comments for any modifications as might be needed.
4. This review cycle continues till the code reaches satisfactory quality.
5. Once ready the code can be merged by an eligible person to the main branch and is ready for release.

The human code reviewers need to ensure the following code styling guidelines are followed. To ensure a thorough and proper review code reviewers should be given at least 1 week before the code needs to be released.

## 4. Reproducibility

The key metric here is that the reviewer should be able to download and run a sample test of your code using a trained model within 15 minutes and without having to go through code. There should be a script to download and prepare data for use with clear and easy instructions. Similarly, there should be a script to run basic experiments from simple tests to complete re-training. Further, the environment needs to be replicable which can be done using the following options:

- Approach 1 (Most basic, not very reliable): requirements.txt file with versions for each package including python

- Approach 2 (Less basic, not reliable): Virtual environments with library installations 

- Approach 3 (Slightly complex, most reliable): Define a docker environment and write a script to prepare the same for runs.

Advantages of docker include choice of OS along with a fixed set of libraries installed. This ensures that the user does not need to install anything on their machine or grapple with versioning issues to be able to run your code.

# Repo Setup Instructions

1. Settings

    - Provide access to the different member groups in the lab as per Settings > Collaborators and teams
    - Add branch protection rule as per Settings > Code and automation > Branch protection rules
