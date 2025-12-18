use crate::plugins::load_registry;

pub fn execute() -> Result<(), Box<dyn std::error::Error>> {
    println!("hodu {}", env!("CARGO_PKG_VERSION"));
    println!("hodu-plugin {}", hodu_plugin::PLUGIN_VERSION);
    println!("Platform: {}", hodu_plugin::current_host_triple());

    // List installed plugins
    println!();
    println!("Installed plugins:");

    match load_registry() {
        Ok(registry) => {
            let backends: Vec<_> = registry.backends().collect();
            let model_formats: Vec<_> = registry.model_formats().collect();
            let tensor_formats: Vec<_> = registry.tensor_formats().collect();

            if backends.is_empty() && model_formats.is_empty() && tensor_formats.is_empty() {
                println!("  (none)");
            } else {
                for plugin in backends {
                    println!("  {} {} [backend]", plugin.name, plugin.version);
                }
                for plugin in model_formats {
                    println!("  {} {} [model_format]", plugin.name, plugin.version);
                }
                for plugin in tensor_formats {
                    println!("  {} {} [tensor_format]", plugin.name, plugin.version);
                }
            }
        },
        Err(_) => println!("  (none)"),
    }

    Ok(())
}
