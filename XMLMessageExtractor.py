from xml.etree import ElementTree as ET

with open('dataset/smsCorpus_en_2014.09.06_all.xml','r') as xmlfile:
    tree = ET.parse(xmlfile)
    root = tree.getroot()

with open('dataset/test/sms-data','w') as textfile:
    for message in root.findall('message'):
        smstext = message.find('text').text
        #print smstext
        line = smstext.encode('utf-8') + '\n'
        textfile.write(line)
    
    textfile.close()
