import re

with open('dataset/test/sms-data','w') as outfile:
    regexp = re.compile('<text>(.+)</text>')
    
    with open('dataset/smsCorpus_en_2014.09.06_all.xml','r') as xmlfile:
        for line in xmlfile:
            line = line.rstrip('\n')
            
            matches = re.findall(regexp,line)
            
            if len(matches) == 0:
                continue
            
            text = matches[0]
            
            if not text.strip():
                continue
            
            line = text + '\n'
            outfile.write(line)
            
        outfile.close()
