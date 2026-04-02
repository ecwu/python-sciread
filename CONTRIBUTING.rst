============
Contributing
============

Contributions are welcome and appreciated. Keep changes scoped,
documented, and covered by tests when behavior changes.

Bug reports
===========

When `reporting a bug <https://github.com/ecwu/python-sciread/issues>`_ please include:

    * Your operating system name and version.
    * Any details about your local setup that might be helpful in troubleshooting.
    * Detailed steps to reproduce the bug.

Documentation improvements
==========================

sciread could always use more documentation, whether as part of the
official sciread docs, in docstrings, or even on the web in blog posts,
articles, and such.

Feature requests and feedback
=============================

The best way to send feedback is to file an issue at https://github.com/ecwu/python-sciread/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that code contributions are welcome :)

Development
===========

To set up `python-sciread` for local development:

1. Fork `python-sciread <https://github.com/ecwu/python-sciread>`_.
2. Clone your fork locally::

    git clone https://github.com/YOUR_USERNAME/python-sciread.git
    cd python-sciread

3. Create a local development environment::

    uv sync --group test --group dev

4. Create a branch for local development::

    git switch -c name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. Run the standard checks before opening a pull request::

    uv run ruff format src/ tests/
    uv run ruff check src/ tests/
    uv run pytest tests/

6. If you changed documentation and want to build it locally::

    uv pip install -r docs/requirements.txt
    uv run sphinx-build -b html docs docs/_build/html

7. Commit your changes and push your branch::

    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through GitHub.

Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code just make the pull request.

For merging, you should:

1. Include passing tests with ``uv run pytest tests/``.
2. Include passing lint checks with ``uv run ruff check src/ tests/``.
3. Run ``uv run ruff format src/ tests/`` after Python changes.
4. Update documentation when there is new API or user-facing behavior.
5. Add a note to ``CHANGELOG.rst`` about the changes.
6. Add or update tests when behavior changes.

Adding yourself to ``AUTHORS.rst`` is welcome for substantial contributions.

Tips
----

To run a subset of tests::

    uv run pytest -k test_myfeature

To run a single test file::

    uv run pytest tests/test_cli.py

To run tests with coverage::

    uv run pytest --cov src/ tests/
