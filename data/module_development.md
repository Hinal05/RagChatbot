# Drupal Custom Module Development Basics

## Minimum files for a module
A custom module named `my_module` needs at least:
- `my_module.info.yml` — declares the module name, type, core_version_requirement, and package.
- `my_module.module` (optional) — contains procedural hook implementations.
- `src/` directory — contains PSR-4 autoloaded classes (Plugins, Controllers, Services, Forms).

## The .info.yml file
```yaml
name: My Module
type: module
description: 'Does something useful.'
core_version_requirement: ^10 || ^11
package: Custom
dependencies:
  - drupal:node
```

## Routing
Routes are declared in `my_module.routing.yml` and map a path to a controller method:
```yaml
my_module.example:
  path: '/my-module/example'
  defaults:
    _controller: '\Drupal\my_module\Controller\ExampleController::content'
    _title: 'Example page'
  requirements:
    _permission: 'access content'
```

## Services and dependency injection
Services are declared in `my_module.services.yml`. Controllers and forms should use dependency injection via `create()` and `ContainerInjectionInterface` rather than calling `\Drupal::service()` directly, to keep code testable.

## Plugins
Drupal's plugin system (Block plugins, Field types, Field widgets, Field formatters, Views plugins) uses annotations or attributes on classes placed under `src/Plugin/{PluginType}/`. Each plugin class must implement the relevant plugin interface.

## Configuration management
- Simple config is stored as YAML under `config/install/` and read via `\Drupal::config('my_module.settings')`.
- Content entities (nodes, users, custom entities) are NOT configuration and are not exported via `drush cex`.

## Caching
- Render arrays should declare `#cache` contexts and tags so the render cache invalidates correctly when the underlying data changes.
- Use `\Drupal::cache()` for arbitrary data caching, and always set appropriate cache tags so entries are invalidated automatically.
