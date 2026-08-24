# Publishing this repository and getting a DOI

Follow these in order. The whole thing takes about thirty minutes, most of it
waiting for uploads.

## Before you start

Replace two placeholders in the files:

    CITATION.cff        repository-code: https://github.com/GITHUB_USERNAME/aegis-iot
    README.md           (no placeholder, but check the paper title matches yours)

Use your real GitHub username in place of `GITHUB_USERNAME`.

## Step 1. Create the repository on GitHub

Go to https://github.com/new and set:

  - Repository name: `aegis-iot`
  - Description: `Accuracy-constrained tool selection for multi-component intrusion detection in IoT networks`
  - Visibility: **Public**
  - Do NOT tick "Add a README", "Add .gitignore" or "Choose a license". This
    package already has all three, and adding them creates a merge conflict on
    your first push.

Click Create repository. Leave the page open; you will need the URL.

## Step 2. Push the code

From inside the unzipped `aegis-iot/` directory:

    git init
    git add .
    git commit -m "Initial release: code, results and figures for the Future Internet submission"
    git branch -M main
    git remote add origin https://github.com/GITHUB_USERNAME/aegis-iot.git
    git push -u origin main

If GitHub asks for a password, it wants a personal access token rather than your
account password. Create one at Settings, Developer settings, Personal access
tokens, Tokens (classic), with the `repo` scope ticked.

Check what you actually pushed:

    git ls-files | wc -l

You should see roughly 109 files. If you see thousands, `.gitignore` is not
being respected and you are about to commit the corpora. Stop, run
`git rm -r --cached raw data`, and commit again.

## Step 3. Connect Zenodo

Zenodo mints the permanent DOI that goes in the paper. GitHub alone does not
give you a citable identifier.

  1. Go to https://zenodo.org and sign in with GitHub.
  2. Click your name, then GitHub.
  3. Find `aegis-iot` in the list and switch the toggle **On**.

Zenodo now watches the repository for releases. It ignores anything pushed
before the toggle was switched on, which is why the next step matters.

## Step 4. Create a release

On your GitHub repository page, click Releases, then "Create a new release".

  - Tag: `v1.0.0`, and choose "Create new tag on publish"
  - Title: `v1.0.0 — Future Internet submission`
  - Description:

        Code, results and figures accompanying the manuscript "When Cost
        Objectives Delete Capability: Accuracy-Constrained Tool Selection for
        Multi-Component Intrusion Detection in IoT Networks", submitted to
        Future Internet (MDPI).

        Includes the full experimental pipeline, the timed cost-measurement
        harness, the results reported in the paper for both corpora, and the
        historical runs in which the failure was first observed.

Click "Publish release".

Wait two or three minutes, then refresh https://zenodo.org/account/settings/github/
Your repository will now show a DOI badge.

## Step 5. Get the two identifiers

Zenodo issues two DOIs and the distinction matters.

  - The **concept DOI** always resolves to the newest version. Use this one.
  - The **version DOI** points at v1.0.0 specifically.

On the Zenodo record page, look for "Cite all versions?" and take the DOI listed
under that. It will look like `10.5281/zenodo.XXXXXXX`.

## Step 6. Put them in the paper

Replace the Data Availability Statement with:

    Data Availability Statement: Both corpora analysed in this study are
    publicly available. The TON_IoT datasets are distributed by UNSW Canberra
    Cyber at https://research.unsw.edu.au/projects/toniot-datasets. This work
    uses train_test_network.csv from the Train_Test_datasets collection. The
    Bot-IoT dataset is available at
    https://research.unsw.edu.au/projects/bot-iot-dataset. This work uses the
    full-feature release. The experimental code, configuration files,
    cost-measurement harness and analysis scripts required to reproduce every
    result reported here are publicly available at
    https://github.com/GITHUB_USERNAME/aegis-iot and archived at
    https://doi.org/10.5281/zenodo.XXXXXXX.

Then add the DOI badge to the top of `README.md`, just under the title:

    [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Commit and push that change. It does not need a new release.

## Step 7. Update the response letter

Reviewer 1 asked specifically whether the code could be made public rather than
available on request. In your point-by-point response, quote the final wording
of the Data Availability Statement with the real URL and DOI filled in, not the
placeholders.

## What not to commit

`raw/` and `data/` are in `.gitignore` and should stay there. The corpora are
redistributed by UNSW under their own terms, they total about sixteen gigabytes,
and GitHub rejects individual files above one hundred megabytes in any case.

If you accidentally commit them, do not simply delete and re-commit: the objects
stay in the history and the repository will still be enormous. Use
`git filter-repo` or start a fresh repository.
