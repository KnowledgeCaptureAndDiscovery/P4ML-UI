# How to release a new version

*A cheat sheet.*

* On `devel` branch:
  * `git pull` to make sure everything is in sync with remote origin.
  * Change a version in `primitive_interfaces/__init__.py` to a version without `rc0`.
  * Change `vNEXT` in `HISTORY.md` to the to-be-released version, with `v` prefix.
  * In `.gitlab-ci.yml` change `DEPENDENCY_REF` to a fixed version tag of dependencies.
  * in `setup.py` remove `d3m_metadata` package entry in the `dependency_links` list,
    or the `dependency_links` list altogether.
  * Commit with message `Bumping version for release.`
  * `git push`
  * Wait for CI to run tests successfully.
* On `master` branch:
  * `git pull` to make sure everything is in sync with remote origin.
  * Merge `devel` into `master` branch: `git merge devel`
  * `git push`
  * Wait for CI to run tests successfully.
  * Release a package to PyPi:
    * `rm -rf dist/`
    * `python setup.py sdist`
    * `twine upload dist/*`
  * Tag with version prefixed with `v`, e.g., for version `2017.9.20`: `git tag v2017.9.20`
  * `git push` & `git push --tags`
* On `devel` branch:
  * `git merge master` to make sure `devel` is always on top of `master`.
  * Change a version in `primitive_interfaces/__init__.py` to the next day (or next known release date) and append `rc0`.
  * Add a new empty `vNEXT` version on top of `HISTORY.md`.
  * In `.gitlab-cy.yml` change `DEPENDENCY_REF` to a development branch of dependencies, probably `devel`.
  * in `setup.py` restore the `d3m_metadata` package entry in the `dependency_links` list,
    ot the `dependency_links` list altogether:

        ```
        dependency_links=[
            'git+https://gitlab.com/datadrivendiscovery/metadata.git@devel#egg=d3m_metadata-{version}'.format(version=version),
        ],
        ```

  * Commit with message `Version bump for development.`
  * `git push`

If there is a need for a patch version to fix a released version on the same day,
use `.postX` prefix, like `2017.9.20.post0`. If more than a day has passed, just
use the new day's version.
