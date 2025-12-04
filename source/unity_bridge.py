import socket
import json
import time
import sys
import traceback

try:
    import emotion_core as core
except ImportError:
    print("Error: emotion_core not found!")
    sys.exit(1)

HOST = '127.0.0.1'
PORT = 65432
FRAME_RATE = 30

def main():
    print("init system")
    if not core.init_system():
        print("Error: Fail to initial system")
        input("Press Enter to exit...")
        return
    print("system ready")

    print(f"waiting for Unity connection({HOST}:{PORT})")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()

        print(f"listening on port:{PORT}")

        while True:
            print(f"\nWating for Unity Connection ({HOST}:{PORT}) ...")

            try:
                conn, addr = s.accept()

                with conn:
                    print(f"Unity connected:{addr}")

                    try:
                        while True:
                            start_time = time.time()

                            # Get Data
                            data = core.get_current_emotion()

                            # Send Data
                            if data:
                                # resize as json
                                json_str = json.dumps(data) + "\n"
                                conn.sendall(json_str.encode('utf-8'))
                                
                                # print log(for debug)
                                if data.get("hasFace"):
                                    print(f"send: {data['label']} (V:{data['valence']:.2f})")
                                else:
                                    print("No face") 
                                    pass
                            
                            # frame control
                            elapsed = time.time() - start_time
                            sleep_time = (1.0 / FRAME_RATE) - elapsed
                            if sleep_time > 0:
                                time.sleep(sleep_time)

                    except (ConnectionResetError, BrokenPipeError):
                        print("Unity Disconnected")
                    except Exception as e:
                        print(f"Error: {e}")
                    finally:
                        print("\nConnection closed. Waiting for next...")

            except KeyboardInterrupt:
                print("\nSystem stoped (Ctrl+C)")
                break
            except Exception as e:
                print(f"\nError: {e}")
                traceback.print_exc()
                time.sleep(1)

    print("Releasing resources...")
    core.release_system()

if __name__ == "__main__":
    main()  
