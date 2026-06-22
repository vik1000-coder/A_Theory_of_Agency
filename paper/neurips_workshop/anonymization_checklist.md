# Anonymization Checklist

Before submitting the blind PDF or supplement:

- Search `paper/neurips_workshop/` for author names, handles, email addresses, local paths, and
  acknowledgements.
- Confirm the PDF title page says `Anonymous Authors`.
- Confirm no public GitHub Pages URL or repository URL appears in the PDF.
- Confirm generated tables do not include absolute paths.
- Confirm supplemental artifact notes describe an anonymous archive rather than the public repo.
- Rerun `build-paper-artifacts` after any new run and inspect `generated/paper_metrics.json` before
  compiling.
