<?php
namespace Rindow\NeuralNetworks\Layer;

class GlobalAveragePooling3D extends AbstractGlobalAveragePooling
{
    protected $rank = 3;

    public function __construct(object $backend,array $options=null)
    {
        $leftargs = [];
        extract($this->extractArgs([
            'name'=>null,
        ],$options,$leftargs));
        $this->initName($name,'globalaveragepooling3d');
        parent::__construct($backend, $leftargs);
    }
}
