"""
Screen Automation Bot Launcher
Simple menu to choose between GUI, CLI, or exit
"""

import os
import sys
import subprocess

def print_menu():
    """Print the main menu"""
    print("\n" + "="*40)
    print("   Screen Automation Bot Launcher")
    print("="*40)
    print("1. GUI Mode (Graphical Interface)")
    print("2. CLI Mode (Command Line Interface)")
    print("3. Exit")
    print("="*40)

def main():
    """Main launcher function"""
    while True:
        print_menu()
        try:
            choice = input("Select an option (1-3): ").strip()

            if choice == "1":
                print("\nLaunching GUI Mode...")
                try:
                    # Run GUI in subprocess
                    subprocess.run([sys.executable, "bot_gui.py"], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error running GUI: {e}")
                    input("Press Enter to continue...")
                except KeyboardInterrupt:
                    print("\n⚠️  GUI interrupted")
                break  # Exit after GUI closes

            elif choice == "2":
                print("\nLaunching CLI Mode...")
                try:
                    # Run CLI in subprocess
                    subprocess.run([sys.executable, "interactive_bot.py"], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error running CLI: {e}")
                    input("Press Enter to continue...")
                except KeyboardInterrupt:
                    print("\n⚠️  CLI interrupted")
                break  # Exit after CLI finishes

            elif choice == "3":
                print("\nGoodbye!")
                sys.exit(0)

            else:
                print("\n❌ Invalid choice. Please select 1, 2, or 3.")

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()