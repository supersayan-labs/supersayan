import sys
from pathlib import Path


def setup_julia_deps():
    """Install Julia dependencies for supersayan."""
    print("Setting up Julia dependencies for supersayan...")

    try:
        # Import juliacall - this will trigger Julia installation if needed
        import juliacall
        from juliacall import Main as jl

        # Get the path to the Julia backend
        julia_backend = Path(__file__).parent / "julia_backend"

        # Check if Project.toml exists
        project_toml = julia_backend / "Project.toml"
        if project_toml.exists():
            print(f"Found Julia project at {julia_backend}")

            # Activate and instantiate the Julia project
            jl.seval(
                f"""
            using Pkg
            Pkg.activate("{julia_backend}")
            Pkg.instantiate()
            println("✓ Julia dependencies installed successfully!")
            """
            )
        else:
            print("Warning: No Project.toml found in julia_backend directory")

    except ImportError:
        print(
            "ERROR: juliacall not found. It should have been installed with supersayan."
        )
        print("Please ensure supersayan was installed correctly.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR setting up Julia dependencies: {e}")
        print("You may need to manually set up Julia dependencies.")
        sys.exit(1)


def _auto_setup():
    """Automatically set up Julia deps when the package is imported."""
    import os

    # Check if we should skip auto-setup (useful for CI/CD)
    if os.environ.get("SUPERSAYAN_SKIP_JULIA_SETUP", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return

    # Check if Julia deps are already set up
    julia_backend = Path(__file__).parent / "julia_backend"
    manifest_file = julia_backend / "Manifest.toml"

    if not manifest_file.exists():
        print("First time setup: Installing Julia dependencies...")
        try:
            setup_julia_deps()
        except SystemExit:
            print("Warning: Failed to set up Julia dependencies automatically.")
            print("Please run 'supersayan-setup' manually.")


if __name__ == "__main__":
    setup_julia_deps()
