# Script that generates signatures for R package for registering native routines.
# The script is run in interactive mode.

# import os
# gpboost_path = # Put path of GPBoost folder here
# os.chdir(gpboost_path)

IN_FILES = ['R-package/src/gpboost_R.h']

functions = {}

for IN_FILE in IN_FILES:
    with open(IN_FILE, 'r') as f:
        txt = [
            line.strip() for line in f
        ]

    # filter lines
    txt = [
        line for line in txt
        if (
            not line.startswith('/')
            and not line.startswith('*')
            and not line.startswith('#')
            and not line == ''
        )
    ]

    prefix = 'GPBOOST_C_EXPORT LGBM_SE '
    current_function = None
    for line in txt:
        if line.startswith(prefix):
            current_function = line.replace(prefix, '').replace('(', '')
            functions[current_function] = 0
            print(current_function)
        elif line.startswith('LGBM_SE'):
            functions[current_function] += 1
        elif line.startswith(');'):
            current_function = None
        else:
            raise RuntimeError(f"Encountered unknown line: '{line}'")

# create output like
# {"LGBM_BoosterSaveModelToString_R", (DL_FUNC) &LGBM_BoosterSaveModelToString_R, 6},
# {"LGBM_BoosterDumpModel_R", (DL_FUNC) &LGBM_BoosterDumpModel_R, 6}
longest_func_length = max([
    len(function_name)
    for function_name in functions.keys()
])
print(f"The longest function is {longest_func_length} characters")
for func_name, num_args in functions.items():
    out = '{"'
    out += func_name
    out += '"'
    out += " " * (longest_func_length - len(func_name))
    out += ', (DL_FUNC) &'
    out += func_name
    out += " " * (longest_func_length - len(func_name))
    out += ', '
    out += str(num_args)
    out += '},'
    print(out)
