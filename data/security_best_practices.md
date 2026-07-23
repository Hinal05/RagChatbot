# Drupal Security Best Practices

## Input sanitization
- Never trust user input. Use Drupal's Form API validation and `#element_validate` callbacks rather than manual checks scattered through business logic.
- When outputting text that may contain HTML from an untrusted source, use `Xss::filter()` or `Xss::filterAdmin()` (only for trusted roles) rather than printing it raw.
- Twig auto-escapes variables by default; only use the `|raw` filter on strings that are already known-safe markup (e.g. output of `#markup` render arrays that were sanitized).

## SQL injection prevention
- Always use the Database API with placeholders: `$connection->query('SELECT * FROM {table} WHERE id = :id', [':id' => $id])`.
- Never build SQL strings by concatenating variables directly into the query string.

## CSRF protection
- State-changing routes (POST/PATCH/DELETE actions triggered by links) should require a CSRF token, which Drupal generates automatically for routes using `_csrf_token: 'TRUE'` in routing requirements.
- Forms built through the Form API get CSRF protection automatically via the `form_token` hidden field.

## Access control
- Every route must declare `requirements` (`_permission`, `_role`, or a custom access check service) — never rely on hiding a link as the only protection.
- Custom access logic should implement `AccessInterface` and be wired via `_custom_access` in routing.yml, so it is enforced even when the route is accessed directly.

## File uploads
- Validate uploaded file extensions using the `file_validate_extensions()` validator or the Form API `#upload_validators`.
- Store uploads outside the public files directory when they should not be directly downloadable without an access check; use Drupal's private file system.

## Third-party libraries
- Keep contributed modules and core updated; security advisories are published on Drupal.org and most vulnerabilities are patched quickly after disclosure.
- Avoid modules with no recent maintenance activity for security-sensitive functionality.

## Secrets management
- Never commit API keys, database credentials, or `.env` files to version control.
- Use environment variables or a settings.local.php (gitignored) for environment-specific secrets, never hardcoded in settings.php committed to the repo.
