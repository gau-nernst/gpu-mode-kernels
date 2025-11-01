import time
import re

def read_and_process_hex_numbers(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
        
        # 使用正则表达式找到所有16进制数字
        hex_pattern = r'0x[0-9a-fA-F]+'
        hex_numbers = re.findall(hex_pattern, content)
        
        if not hex_numbers:
            print("文件中没有找到16进制数字")
            return
        
        # 转换为整数
        int_numbers = [int(hex_num, 16) for hex_num in hex_numbers]
        
        # 找到最小值
        min_value = min(int_numbers)
        min_hex = hex(min_value)
        
        print(f"找到 {len(hex_numbers)} 个16进制数字")
        print(f"最小值: {min_hex} ({min_value})")
        print("各数字相对于最小值的偏移量:")
        print("-" * 50)
        
        for i, (hex_num, int_num) in enumerate(zip(hex_numbers, int_numbers)):
            offset = int_num - min_value
            print(f"Index {i}: {hex_num} -> 偏移量: {offset} (0x{offset:x})")
        
        print("-" * 50)
        
    except FileNotFoundError:
        print(f"文件 {filename} 不存在")
    except Exception as e:
        print(f"处理文件时出错: {e}")

def main():
    filename = "a.txt"
    
    print("开始监控文件，每秒读取一次...")
    print("按 Ctrl+C 停止程序")
    
    try:
        while True:
            print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
            read_and_process_hex_numbers(filename)
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n程序已停止")

if __name__ == "__main__":
    main()
