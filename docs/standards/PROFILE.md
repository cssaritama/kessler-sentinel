# Project profile: `product`

Version 2.6.0. Adds to the [project standards](PROJECT_STANDARDS.md) and never relaxes them.
Default data class: **Internal** ([Data classification](DATA_CLASSIFICATION.md)).

## Rules every profile shares

1. The entry point is `CONTRIBUTING.md`. It names the profile, the owner, the data
   class and the lifecycle state: Proposed, Active, Maintenance or Archived.
2. The active specification follows the [specification template](templates/SPEC_TEMPLATE.md)
   or records why a project convention replaces it.
3. A claim of a result links a verification record based on the
   [verification template](templates/VERIFICATION_TEMPLATE.md).

## Rules for `product`

1. Each release has a version, release notes and a verification record.
2. User-facing acceptance is written as examples a reviewer can run.
3. A breaking change to an interface states its migration path before release.
