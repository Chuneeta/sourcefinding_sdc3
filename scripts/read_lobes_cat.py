from argparse import ArgumentParser
import numpy as np
import pylab
import yaml
import time

parser=ArgumentParser(description='reading pyaml file')
parser.add_argument('fl', type=str, help='Name of input catalogue')
parser.add_argument('--outfile', type=str, help='Name of output catalogue', default='lobes_psource_catalogue')
args = parser.parse_args(

start_time = time.perf_counter()

with open(args.fl, "r") s h:
	source_list = yaml.load(h, Loader=yaml.SafeLoader)

end_time = time.perf_counter ()
print("The loading of yaml files takes:", end_time - start_time, "seconds")

print("Number of sources in the source list: {}".format(len(source_list)))
S = np.zeros((len(source_list), 5), dtype=(object))
print("Iterating over sources")
count = 0 
N_src = 0
for source_name in sourcie_list:
    if source_name[:3] == 'SNG':
        print("Source name: {}".format(source_name))
        for i, component in enumerate(source_list[source_name]):     
            if "point" in component["comp_type"]:
                # Do stuff with point components here.
                S[count, 0] = source_name
                S[count, 1] =  component["ra"]
                S[count, 2] =  component["dec"]
     
            if "power_law" in component["flux_type"]:
                pl = component["flux_type"]["power_law"]
                S[count, 3] = pl["fd"]['freq']
                S[count, 4] = pl["fd"]['i']
           
            count = count +1
        print("------------------------------------------------")

        N_src+= 1    

S = np.delete(S, np.where(S[:,3] == 0)[0], axis=0) # to remove the rows where the frequency is not measured.
S = np.delete(S, np.where(S[:,0] == 0)[0], axis=0)
S1 = np.delete(S, np.where(S[:,4] < 1)[0], axis=0)
print("Number of interested sources in the source list: {}".format(len(S1)))
np.save(args.outfile, S1)
