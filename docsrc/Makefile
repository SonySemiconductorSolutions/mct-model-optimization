# Minimal makefile for Sphinx documentation
#

github:
	@rm -rf ../docs/
	@mkdir ../docs
	@mkdir ../docs/docs_troubleshoot/
	@make html
	@cp -a ../docs/html/. ../docs
	@cp -a ../docs/docs_troubleshoot/html/. ../docs/docs_troubleshoot/
	@rm -rf ../docs/html/
	@rm -rf ../docs/doctrees/
	@rm -rf ../docs/docs_troubleshoot/html/
	@rm -rf ../docs/docs_troubleshoot/doctrees/
	@find ../docs -type f -exec sed -i 's|_images/|images/|g' {} +
	@find ../docs -type f -exec sed -i 's|_static/|static/|g' {} +
	@find ../docs/docs_troubleshoot -type f -exec sed -i 's|_images/|images/|g' {} +
	@find ../docs/docs_troubleshoot -type f -exec sed -i 's|_static/|static/|g' {} +
	@rm -rf ../docs/images
	@rm -rf ../docs/static
	@rm -rf ../docs/docs_troubleshoot/images
	@rm -rf ../docs/docs_troubleshoot/static
	@mv ../docs/_images/ ../docs/images
	@mv ../docs/_static/ ../docs/static
	@mv ../docs/docs_troubleshoot/_images/ ../docs/docs_troubleshoot/images
	@mv ../docs/docs_troubleshoot/_static/ ../docs/docs_troubleshoot/static



# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = ../docs

SPHINXOPTS_TROUBLESHOOT    ?=
SPHINXBUILD_TROUBLESHOOT   ?= sphinx-build
SOURCEDIR_TROUBLESHOOT     = source_troubleshoot
BUILDDIR_TROUBLESHOOT      = ../docs/docs_troubleshoot

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@$(SPHINXBUILD_TROUBLESHOOT) -M $@ "$(SOURCEDIR_TROUBLESHOOT)" "$(BUILDDIR_TROUBLESHOOT)" $(SPHINXOPTS_TROUBLESHOOT) $(O)