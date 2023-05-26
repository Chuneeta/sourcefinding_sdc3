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
Nsrc = 0
for source_name in source_list:
    for i, component in enumerate(source_list[source_name]):
        if component['comp_type'] == 'point':
            flux_list = component['flux_type']['list']
            for j in range(len(flux_list)):
                if flux_list[j]['freq'] == 152e6:
                    S[count, 0] = source_name
                    S[count, 1] =  component["ra"]
                    S[count, 2] =  component["dec"]
                    S[count, 3] = flux_list['freq']
                    S[count, 4] = flux_list['i']
                    count += 1
                    Nsrc += 1
   
        print("------------------------------------------------")

S = np.delete(S, np.where(S[:,0] == 0)[0], axis=0)
np.save(args.outfile, S)
