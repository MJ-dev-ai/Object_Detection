from config import flags
import time, datetime

def main():
    start_time = time.time()
    
    # Simulate some processing
    time.sleep(5)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Main completed in: {str(datetime.timedelta(seconds=elapsed_time))}")

if __name__ == "__main__":
    main()