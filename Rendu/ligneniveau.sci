//paraboloide elliptique : graphe
x=-8:0.1:8;y=-8:0.1:8;plot3d(x,y,((x^2)')*ones(x)+ones(y')*(y^2),theta=75,alpha=45,flag=[2,2,0])

//paraboloide elliptique : lignes de niveaux
function z=selle(x,y)
  z=x^2+y^2
  endfunction
  x=[-6:0.05:6]';
  y=x;
clf()  
  z=feval(x,y,selle); 
  grayplot(x,y,z)
  f=gcf()//handle figure
  f.color_map=hotcolormap(64)
  Sgrayplot(x,y,z)
  
clf()
  f=gcf()
  f.color_map=hotcolormap(32)
  colorbar(-1,1) 
  Sgrayplot(x,y,z)
  contour(x,y,z,5)