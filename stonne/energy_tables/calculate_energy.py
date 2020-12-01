#! /usr/bin/env python

import sys
import re
import os.path
from os import path

## Read the arguments that the user introduces.
# @return verbose enables verbose mode.
# @return table_file the path to the table with the energy numbers of each element.
# @return counter_file The path to the counter file with the number of operations performed for each element.
# @return the output file where prints the energy numbers. 
def read_arguments():
    verbose=False
    table_file=""
    counter_file=""
    out_file=""
    for i in sys.argv[1:]:
        # Arguments that do not require = symbol
        if(i=="-v"):
            verbose=True

        # Arguments that DO require = symbol
        splited=i.split("=")
        if(splited[0]=="-table_file"):
            table_file=splited[1]

        elif(splited[0]=="-counter_file"):
            counter_file=splited[1]

        elif(splited[0]=="-out_file"):
            out_file=splited[1]

    #Check if the user has introduced the parameters correctly
    if((table_file=="") or (counter_file=="")):
        print("Error parsing input parameters")
        print("Usage: ./"+str(sys.argv[0]+" [-v] -table_file=<Energy numbers file> -counter_file=<Runtime counters file> -[out_file=<output file>]"))
        exit(2)

    #Checking if table_file exists
    if(not path.isfile(table_file)):
        print("Error file "+table_file+" does not exist")
        exit(2)

    #Checking if counter_file exists
    if(not path.isfile(counter_file)):
        print("Error file "+counter_file+" does not exist")
        exit(2)

    # If out_file is not an user parameter then out_file=counter_file.energy
    if(out_file==""):
        out_file=counter_file+".energy"
    
    return verbose, table_file, counter_file, out_file

## Reads an input file with the table of energy/operation and store it in a dictionary
# @param table_file is the file with the table 
# @param out_file is the out_file to print the table there
#  @return energy_table a dictionary whose keys are the components and every key corresponds with another dictionary 
#  whose keys are the operations of that component. The value of every entry is the energ/operation of a certain component
#  and a certain operation over that component. Ej: energy_table[GLOBALBUFFER][READ]=2
def getEnergyTable(table_file, out_file):
    energy_table={}  # Dynamic energy table
    f = open(table_file, "r")   # We are sure it exists
    o = open(out_file, "w+")    # We create the file if it does not exist
    
    #Writing useful information in output file
    o.write("DYNAMIC ENERGY TABLE USED\n")
    o.write("-------------------------------------------\n")
    line_number=0
    #Generating the table 
    for line in f:
        line_number+=1
        #Printing the table into the output file
        if(line != "\n"): #newlines are ignored
            o.write(line)

        # Proceed with the code that generates the table for that component and operation.
        line_splited=line.rstrip().split(' ')
        if(len(line_splited)>=1):
            component=line_splited[0]
            energy_table[component]={} # Creating the dictionary for this component
            for op in line_splited[1:]:
                op_splited=op.split('=')
                if(len(op_splited) != 2):
                    print('Error parsing line '+str(line_number)+": "+op+" is not recognized")
                    exit(3)

                operation=op_splited[0]
                energy=float(op_splited[1])
                energy_table[component][operation]=energy #Inserting key=operation value=energy

    # Printing more useful characters into the output file
    o.write("-------------------------------------------\n")
    # Closing the files
    f.close()
    o.close()
    return energy_table

## Calculates the total dynamic energy consumption and store the result in out_file. If verbose is enabled the function 
#  prints the dynamic energy consumed by every element/operation. If not, only high level components and the total is 
#  printed
#  @energy_table dictionary energy_table[component][operation]=energy that consumes the certain component when certain 
#  operation is carried out
#  @counter_file the path of a file with a specific format that shows for every component/operation the number of 
#  operations performed  during a certain simulation
#  @out_file the path of the output file

def calculateEnergy(energy_table, counter_file, out_file, verbose):
    cf=open(counter_file, "r")       # We are sure it exists
    out_file=open(out_file, "a")    # We are sure it was created in getEnergyTable
    line_number=0
    dynamic_energy_global=float(0.0)            # Global dynamic energy 
    static_energy_global=float(0.0)             # Global static energy
    area_global=float(0.0)
    dynamic_energy_component={}                 # Dictionary with the dynamic energy of every high level compo
    static_energy_component={}                   # Dictionary with the static energy of every high level compo
    area_component={}                            # Dictionary with the area of every high level cmponent
    current_component=""                        # String with the current high level component according to []
    #Exctracting general information such us the number of cycles to multiply by the static energy
    first_line=cf.readline().split('=')
    if((len(first_line)>0) and (first_line[0]=="CYCLES")):
        number_of_cycles=int(first_line[1])
    else:
        number_of_cycles=0
        cf.close()
        # If the first line is not CYCLES, we open the file again to start reading from the beginning. 
        # Not doing this would skip the first line, which could be wrong if CYCLES is ommited.
        # Note that we do this since CYCLES is not mandatory in the runtime file
        cf=open(counter_file, "r") 


    for line_counter in cf:
        line_number+=1
        # Check if it is a new high level component
        line_counter_clean = line_counter.rstrip()
        if(line_counter_clean != ""): # Ignore new and void lines
            if ((line_counter_clean[0]=="[") and (line_counter_clean[len(line_counter_clean)-1]=="]")):
                if(verbose):  # If verbose we are going to print every component of every high level component
                    out_file.write(line_counter)
                #Updating current component
                current_component=line_counter_clean[1:len(line_counter_clean)-1]
                dynamic_energy_component[current_component]=float(0.0)
                static_energy_component[current_component]=float(0.0)
                area_component[current_component]=float(0.0)
            else:  # If it is a line with a counter, then, we calculate the energy of that operation
                line_counter_splited = line_counter_clean.split(' ')
                if(len(line_counter_splited)==0):
                    print('Error to parse line '+str(line_number))
                    exit(3)

                component=line_counter_splited[0] # Saving the component
                # We check if the component exists according to out energy table
                if(component not in energy_table):
                    print('Error to parse line '+str(line_number)+': Component '+component+' does not exist in the table')
                    exit(3)
                # If the component exists, we proceed to calculate the energy
                #STATIC ENERGY
                # Looking up in the energy table if the component has a static energy value associated
                # Note that we use STATIC as if it were an operation in the energy table, but actually
                # is the value of the static energy per cycle of that element
                if ("STATIC" in energy_table[component]):
                    static_energy=float(float(number_of_cycles)*float(energy_table[component]["STATIC"]))
                else:
                    static_energy=float(0.0)
                if(verbose):
                    out_file.write(component+" STATIC="+str(static_energy))
                # Accumulating static energy
                static_energy_component[current_component]+=static_energy
                static_energy_global+=static_energy
                # We do the same process for the area
                if("AREA" in energy_table[component]):
                    area=float(energy_table[component]["AREA"])
                else:
                    area=float(0.0)
                if(verbose):
                    out_file.write(" AREA="+str(area))
                #Accumulating the area. TODO: Note that in this version we just do a simple addition of the component
                area_component[current_component]+=area
                area_global+=area
                # DYNAMIC ENERGY. Searching for each operation in that component
                for op in line_counter_splited[1:]:
                #Checking if the operation syntaxis is correct
                    op_splited=op.split('=') # Separating the operation and the counter
                    if(len(op_splited) != 2):
                        print('Error to parse line '+str(line_number)+": "+op+" not recognized")
                        exit(3)
                    # else, we check if the component and operation exists according to our energy table
                    operation=op_splited[0]
                    counter=int(op_splited[1])
                    # Checking if the operation exists
                    if(operation not in energy_table[component]):
                        print('Error to parse line '+str(line_number)+': Operation '+operation+'does not exist in the table.')
                        exit(3)

                    #If everything exists then is a valid operation and can be used to calculate the energy
                    dynamic_energy=float(float(counter)*float(energy_table[component][operation]))
                    # If verbose then we print the energy of that operation instead of counter
                    if(verbose):
                        out_file.write(" "+operation+"="+str(dynamic_energy))
                    #accumulate 
                    dynamic_energy_component[current_component]+=dynamic_energy
                    dynamic_energy_global+=dynamic_energy
                # After the loop, if lines has been inserted per every element, we print the new line
                if(verbose):
                    out_file.write("\n")

            
    #End for                
    # At the end of the loop we print the general statistics
    out_file.write("COMPONENT BREAKDOWN\n")
    out_file.write("-------------------------------------------\n")
    for key in dynamic_energy_component:
        out_file.write(key+": STATIC="+str(static_energy_component[key]))
        out_file.write(" AREA="+str(area_component[key]))
        out_file.write(" DYNAMIC="+str(dynamic_energy_component[key])+"\n") # Note that they both share keys
    out_file.write("Total STATIC Energy: "+str(static_energy_global)+"\n") 
    out_file.write("Total DYNAMIC Energy: "+str(dynamic_energy_global)+"\n")
    total_energy=float(dynamic_energy_global+static_energy_global)
    out_file.write("Total Area: "+str(area_global)+"\n")
    out_file.write("Total Energy: "+str(total_energy))

    #Closing the files
    out_file.close()
    cf.close()

def main():
    #Reading input parameters
    verbose, table_file, counter_file, out_file = read_arguments()
    #Get a dictionary with the values of energy/operation
    energy_table=getEnergyTable(table_file, out_file)
    #Generate the  energy using the counter file and the table previously generated
    calculateEnergy(energy_table, counter_file, out_file, verbose)
    print(verbose)
    print(table_file)
    print(counter_file)
    print(out_file)
    print(energy_table)

#Running the main function
if __name__== "__main__":
   main()

