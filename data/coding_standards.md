# Drupal PHP Coding Standards

Drupal follows a documented coding standard for all PHP code contributed to core and contributed modules.

## Indentation and whitespace
- Use 2 spaces for indentation, never tabs.
- Lines should not have trailing whitespace.
- Files must end with a single newline character.

## Naming conventions
- Function names use lowercase with underscores: `my_module_do_something()`.
- Class names use UpperCamelCase: `MyServiceClass`.
- Module machine names use lowercase with underscores only, no hyphens: `my_module`.
- Hook implementations are named `hook_NAME()`, e.g. `hook_form_alter()`, and in a module named `my_module` this becomes `my_module_form_alter()`.

## Control structures
- Always use curly braces for `if`, `for`, `foreach`, `while`, even for single-line bodies.
- Put a space before the opening parenthesis in control structures: `if (condition) {`.
- `else` and `elseif` go on their own line, not on the closing brace line (unlike some other PHP style guides).

## Comments and documentation
- All functions, classes, and methods must have a docblock following Drupal's API documentation standards (`@param`, `@return`, `@throws`).
- Inline comments should explain "why", not "what" — the code already shows what it does.

## Arrays
- Use short array syntax `[]` instead of `array()`.
- Multi-line arrays should have a trailing comma after the last element.

## Security-related standards
- Never concatenate user input directly into SQL queries; always use the database API's placeholders.
- Always run render output through Twig's autoescaping or `Xss::filter()` when outputting user-provided text outside of Twig.
- Use `#markup` only for trusted, sanitized content; prefer render arrays and `#plain_text` for untrusted strings.
