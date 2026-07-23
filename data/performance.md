# Drupal Performance Tuning

## Caching layers
- Page cache (for anonymous users) and dynamic page cache (for authenticated users) are enabled by default in Drupal 9/10/11 core.
- Render caching stores rendered output of render arrays keyed by cache contexts (e.g. per-user, per-language) and invalidated by cache tags (e.g. `node:123`).
- Internal Page Cache should sit behind a reverse proxy (Varnish, or a CDN) in production for anonymous traffic.

## Avoiding N+1 queries
- When rendering a list of entities, load them all with `\Drupal::entityTypeManager()->getStorage('node')->loadMultiple($ids)` rather than loading one at a time in a loop.
- Preprocess functions that call `\Drupal::entityTypeManager()` or run custom queries per-item in a loop are a common source of N+1 problems on listing pages.

## BigPipe and lazy loading
- BigPipe (in core) allows personalized parts of a page to stream in after the cacheable shell has already been sent to the browser, improving perceived performance for authenticated users.

## Asset aggregation
- CSS and JS aggregation (Configuration > Development > Performance) should be enabled in production to reduce the number of HTTP requests.
- Use `#attached` libraries defined in `my_module.libraries.yml` rather than manually adding `<script>`/`<link>` tags, so Drupal can aggregate and version them correctly.

## Database query performance
- Add indexes for frequently queried fields, especially in custom entity queries and Views filters.
- Use `EntityQuery`/`\Drupal::entityQuery()` rather than raw SQL where possible so query alterations (e.g. access checks, multilingual) apply consistently.

## Cron and queues
- Long-running or high-volume tasks (bulk emails, image processing, imports) should use the Queue API and be processed during cron rather than blocking a user-facing request.
