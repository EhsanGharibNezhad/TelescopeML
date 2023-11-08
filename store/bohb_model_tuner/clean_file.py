input_file = 'out__cnn'
output_file = 'cleaned_out__cnn'




with open(input_file, 'r') as file:
    lines = file.readlines()

filtered_lines = [line for line in lines if not (line.startswith('DEBUG') or line.startswith('INFO'))]

with open(output_file, 'w') as file:
    file.writelines(filtered_lines)






