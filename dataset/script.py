def remove_extension(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if line.endswith('.jpg'):
                line = line[:-4]
            f_out.write(line + '\n')

# Apply the function to your files
remove_extension('train.txt', 'train_id.txt')
remove_extension('test.txt', 'test_id.txt')
remove_extension('val.txt', 'val_id.txt')