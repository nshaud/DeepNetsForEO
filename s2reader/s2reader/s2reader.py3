--- s2reader.py	(original)
+++ s2reader.py	(refactored)
@@ -259,7 +259,7 @@
             root = self._metadata.getroot()
         return {
             k: v
-            for k, v in root.nsmap.iteritems()
+            for k, v in root.nsmap.items()
             if k
         }
 
@@ -444,7 +444,7 @@
             root = fromstring(self.dataset._zipfile.read(gml))
         else:
             root = parse(gml).getroot()
-        nsmap = {k: v for k, v in root.nsmap.items() if k}
+        nsmap = {k: v for k, v in list(root.nsmap.items()) if k}
         try:
             for mask_member in root.iterfind(
                     "eop:maskMembers", namespaces=nsmap):
